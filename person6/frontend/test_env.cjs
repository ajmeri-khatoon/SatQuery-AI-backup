const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  
  try {
    await page.goto('http://localhost:5173/register');
    
    // Evaluate API_BASE_URL in the context of the page
    // Since it's bundled, we might not have access to it globally.
    // Let's just look at the DOM or see if there's any network request.
    
    // Fill in the form
    await page.type('#email', 'test_browser2@example.com');
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
