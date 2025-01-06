FROM node:23-slim

WORKDIR /app

COPY main.js .

ARG EXTRA_REQUIREMENTS
RUN if [ -n "$EXTRA_REQUIREMENTS" ]; then npm install $EXTRA_REQUIREMENTS; fi

ENTRYPOINT ["node", "main.js"]
