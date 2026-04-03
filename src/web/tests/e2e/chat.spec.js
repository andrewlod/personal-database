// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Personal Database Chat', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/query/', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          answer: 'This is a test answer from the knowledge base.',
          sources: [
            {
              title: 'Test Document',
              score: 0.95,
              document_id: 'test-doc-12345',
              content_preview: 'This is a preview of the test document.',
            },
          ],
          processing_time_seconds: 1.5,
        }),
      });
    });
    await page.goto('/');
  });

  test('displays the empty state message on load', async ({ page }) => {
    await expect(
      page.getByText('Start by asking a question about your personal knowledge base.')
    ).toBeVisible();
  });

  test('has a text input field and send button', async ({ page }) => {
    await expect(
      page.getByPlaceholder('Ask a question about your knowledge base...')
    ).toBeVisible();
    await expect(page.getByRole('button', { name: /send/i })).toBeVisible();
  });

  test('has a clear chat button', async ({ page }) => {
    await expect(page.getByRole('button', { name: /clear chat/i })).toBeVisible();
  });

  test('send button is disabled when input is empty', async ({ page }) => {
    const sendButton = page.getByRole('button', { name: /send/i });
    await expect(sendButton).toBeDisabled();
  });

  test('send button is enabled when input has text', async ({ page }) => {
    const input = page.getByPlaceholder('Ask a question about your knowledge base...');
    await input.fill('Hello world');
    const sendButton = page.getByRole('button', { name: /send/i });
    await expect(sendButton).toBeEnabled();
  });

  test('send button is disabled when input is only whitespace', async ({ page }) => {
    const input = page.getByPlaceholder('Ask a question about your knowledge base...');
    await input.fill('   ');
    const sendButton = page.getByRole('button', { name: /send/i });
    await expect(sendButton).toBeDisabled();
  });

  test('can send a message and receive a response', async ({ page }) => {
    const input = page.getByPlaceholder('Ask a question about your knowledge base...');
    await input.fill('What is this?');

    await page.getByRole('button', { name: /send/i }).click();

    await expect(page.getByText('What is this?')).toBeVisible();
    await expect(page.getByText('This is a test answer from the knowledge base.')).toBeVisible();
  });

  test('input is cleared after sending a message', async ({ page }) => {
    const input = page.getByPlaceholder('Ask a question about your knowledge base...');
    await input.fill('What is this?');
    await page.getByRole('button', { name: /send/i }).click();

    await expect(input).toHaveValue('');
  });

  test('can send multiple messages in a conversation', async ({ page }) => {
    const input = page.getByPlaceholder('Ask a question about your knowledge base...');

    await input.fill('First question');
    await page.getByRole('button', { name: /send/i }).click();
    await expect(page.getByText('First question')).toBeVisible();
    await expect(page.getByText('This is a test answer from the knowledge base.')).toBeVisible();

    await input.fill('Second question');
    await page.getByRole('button', { name: /send/i }).click();
    await expect(page.getByText('Second question')).toBeVisible();
  });

  test('clear chat removes all messages', async ({ page }) => {
    const input = page.getByPlaceholder('Ask a question about your knowledge base...');
    await input.fill('Test question');
    await page.getByRole('button', { name: /send/i }).click();

    await expect(page.getByText('Test question')).toBeVisible();
    await expect(page.getByText('This is a test answer from the knowledge base.')).toBeVisible();

    await page.getByRole('button', { name: /clear chat/i }).click();

    await expect(page.getByText('Test question')).not.toBeVisible();
    await expect(
      page.getByText('Start by asking a question about your personal knowledge base.')
    ).toBeVisible();
  });

  test('displays sources in the response', async ({ page }) => {
    const input = page.getByPlaceholder('Ask a question about your knowledge base...');
    await input.fill('What is this?');
    await page.getByRole('button', { name: /send/i }).click();

    await expect(page.getByText('Sources (1):')).toBeVisible();
    await expect(page.getByText('Test Document', { exact: true }).first()).toBeVisible();
    await expect(page.getByText(/Relevance: 95%/)).toBeVisible();
  });

  test('displays processing time in the response', async ({ page }) => {
    const input = page.getByPlaceholder('Ask a question about your knowledge base...');
    await input.fill('What is this?');
    await page.getByRole('button', { name: /send/i }).click();

    await expect(page.getByText('(1.5s)')).toBeVisible();
  });
});
