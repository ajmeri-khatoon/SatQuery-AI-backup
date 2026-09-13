const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  
  page.on('request', request => {
    console.log('REQUEST:', request.method(), request.url());
    console.log('HEADERS:', request.headers());
  });
  
  page.on('response', response => {
    console.log('RESPONSE:', response.status(), response.url());
    console.log('HEADERS:', response.headers());
  });

  try {
    await page.goto('http://localhost:5173/register');
    
    // Fill in the form
    await page.type('#email', 'test_browser@example.com');
    await page.type('#password', 'password123');
    
    // Click submit
    await Promise.all([
      page.click('button[type="submit"]'),
      page.waitForNetworkIdle()
    ]);
  } catch (err) {
    console.error(err);
  } finally {
    await browser.close();
  }
})();
