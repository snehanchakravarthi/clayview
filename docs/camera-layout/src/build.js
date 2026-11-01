// Inlines model.js into template.html and writes ../planner.html, a standalone page.
const fs = require('fs');
const path = require('path');

const page = fs.readFileSync(path.join(__dirname, 'template.html'), 'utf8');
const model = fs.readFileSync(path.join(__dirname, 'model.js'), 'utf8');
const body = page.replace('/*MODEL*/', () => model);
const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head><body style="margin:0">${body}</body></html>`;
fs.writeFileSync(path.join(__dirname, '..', 'planner.html'), html);
