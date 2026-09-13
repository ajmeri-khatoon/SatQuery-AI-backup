const puppeteer = require("puppeteer");

async function run() {
  const browser = await puppeteer.launch({ headless: "new" });
  const page = await browser.newPage();
  
  // Go to Maps page
  await page.goto("http://localhost:5173/maps", { waitUntil: "networkidle0" });
  
  const title = await page.$eval("h2", el => el.textContent);
  console.log("Maps Page Title:", title);
  
  await browser.close();
}

run().catch(console.error);
