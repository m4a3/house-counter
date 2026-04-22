import * as core from "@actions/core";
import * as github from "@actions/github";
import { Client } from "@notionhq/client";

function extractTaskId(text) {
  if (!text) return null;
  const regex = /(fixes|closes|refs)\s+(MNA-\d+)/i;
  const match = text.match(regex);
  return match ? match[2].toUpperCase() : null;
}

function getStatusName(page, statusPropertyName) {
  const prop = page?.properties?.[statusPropertyName];
  if (!prop) return null;
  if (prop.type === "status") return prop.status?.name ?? null;
  if (prop.type === "select") return prop.select?.name ?? null;
  return null;
}

function listPageProperties(page) {
  return Object.entries(page?.properties ?? {})
    .map(([name, p]) => `  • "${name}" (${p.type})`)
    .join("\n");
}

function resolveDataSourceId(db) {
  const sources = db?.data_sources;
  if (!Array.isArray(sources) || sources.length === 0) return null;
  return sources[0].id;
}

async function fetchSamplePage(notion, db, databaseId) {
  const dataSourceId = resolveDataSourceId(db);
  if (dataSourceId && notion.dataSources?.query) {
    const res = await notion.dataSources.query({ data_source_id: dataSourceId, page_size: 1 });
    return res?.results?.[0] ?? null;
  }
  if (notion.databases?.query) {
    const res = await notion.databases.query({ database_id: databaseId, page_size: 1 });
    return res?.results?.[0] ?? null;
  }
  return null;
}

function detectFilterType(samplePage, propertyName) {
  const prop = samplePage?.properties?.[propertyName];
  if (!prop) return null;
  if (prop.type === "title") return "title";
  if (prop.type === "unique_id") return "unique_id";
  return "rich_text";
}

async function queryFiltered(notion, db, databaseId, taskIdProperty, filterType, taskId) {
  const filter =
    filterType === "unique_id"
      ? { property: taskIdProperty, unique_id: { equals: parseInt(taskId.replace(/\D/g, ""), 10) } }
      : { property: taskIdProperty, [filterType]: { equals: taskId } };

  const dataSourceId = resolveDataSourceId(db);
  if (dataSourceId && notion.dataSources?.query) {
    return notion.dataSources.query({ data_source_id: dataSourceId, filter, page_size: 1 });
  }
  if (notion.databases?.query) {
    return notion.databases.query({ database_id: databaseId, filter, page_size: 1 });
  }
  core.setFailed("❌ Unsupported Notion SDK: neither dataSources.query nor databases.query is available.");
  return null;
}

async function run() {
  try {
    const notionToken = process.env.NOTION_TOKEN;
    const databaseId = process.env.NOTION_DATABASE_ID;

    const taskIdProperty = process.env.NOTION_TASK_ID_PROPERTY || "ID";
    const statusProperty = process.env.NOTION_STATUS_PROPERTY || "Status";
    const blockedStatuses = (process.env.NOTION_BLOCKED_STATUSES || "Done,Completed")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);

    if (!notionToken) return core.setFailed("Missing NOTION_TOKEN secret.");
    if (!databaseId) return core.setFailed("Missing NOTION_DATABASE_ID secret.");

    const pr = github.context.payload.pull_request;
    if (!pr) return core.setFailed("This action must run on pull_request events.");

    const taskId = extractTaskId(pr.body || "");
    if (!taskId) {
      return core.setFailed("❌ PR must include a task reference like: fixes MNA-1234");
    }

    const notion = new Client({ auth: notionToken });
    const db = await notion.databases.retrieve({ database_id: databaseId });

    const sample = await fetchSamplePage(notion, db, databaseId);
    if (!sample) {
      return core.setFailed("❌ Database appears empty — cannot discover properties.");
    }

    const availableProps = listPageProperties(sample);
    const sampleProps = sample.properties ?? {};

    if (!sampleProps[taskIdProperty]) {
      return core.setFailed(
        `❌ Property "${taskIdProperty}" (NOTION_TASK_ID_PROPERTY) not found.\nAvailable properties:\n${availableProps}`
      );
    }
    if (!sampleProps[statusProperty]) {
      core.warning(
        `Property "${statusProperty}" (NOTION_STATUS_PROPERTY) not found — status check will be skipped.\nAvailable properties:\n${availableProps}`
      );
    }

    const filterType = detectFilterType(sample, taskIdProperty);
    core.info(`Querying "${taskIdProperty}" (${filterType}) = "${taskId}"`);

    const response = await queryFiltered(notion, db, databaseId, taskIdProperty, filterType, taskId);
    if (!response) return;

    if (!response.results || response.results.length === 0) {
      return core.setFailed(`❌ No Notion task found with ${taskIdProperty} = ${taskId}`);
    }

    const page = response.results[0];
    const status = getStatusName(page, statusProperty);

    if (status && blockedStatuses.includes(status)) {
      return core.setFailed(`❌ Task ${taskId} is "${status}" (blocked). Link an active task.`);
    }

    core.notice(`✅ Valid Notion task found: ${taskId}${status ? ` | Status: ${status}` : ""}`);
  } catch (err) {
    core.setFailed(err instanceof Error ? err.message : String(err));
  }
}

run();
