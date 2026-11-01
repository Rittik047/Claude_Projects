// Render guide.html to PDF with headless Chromium (Playwright). Usage: node render_pdf.js in.html out.pdf
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const [inp, out] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + path.resolve(inp), { waitUntil: 'load' });
  await page.pdf({
    path: out, format: 'A4', printBackground: true, preferCSSPageSize: true,
    displayHeaderFooter: true, headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:7px;width:100%;text-align:center;color:#8a94a6;' +
      'font-family:Inter,sans-serif">ecg-trust · project guide · page <span class="pageNumber">' +
      '</span> / <span class="totalPages"></span></div>',
  });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
