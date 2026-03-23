FROM nginx:alpine

ARG APP_VERSION=unknown
ARG BUILD_DATE=unknown

COPY nginx.conf /etc/nginx/nginx.conf
RUN echo "{\"application\":\"house-counter\",\"version\":\"${APP_VERSION}\",\"build_date\":\"${BUILD_DATE}\"}" > /usr/share/nginx/html/version.json

EXPOSE 80
