import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { lightTheme, darkTheme } from '../theme';
import { ThemeContext } from '../ThemeContext';
import axios from 'axios';
import App from '../App';

/* eslint-disable testing-library/no-node-access */

jest.mock('axios');
jest.mock('react-markdown', () => {
  return function MockMarkdown({ children }) {
    const r = require('react');
    return r.createElement('div', { 'data-testid': 'markdown' }, children);
  };
});

const mockAxiosPost = axios.post;

const submitForm = (container) => {
  const form = container.querySelector('form');
  fireEvent.submit(form);
};

const renderWithTheme = (theme, ui, mode = 'dark') => {
  return render(
    <ThemeContext.Provider value={{ mode, toggleTheme: jest.fn() }}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {ui}
      </ThemeProvider>
    </ThemeContext.Provider>
  );
};

describe('App', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Initial render', () => {
    it('renders the empty state message', () => {
      renderWithTheme(darkTheme, <App />);
      expect(
        screen.getByText('Start by asking a question about your personal knowledge base.')
      ).toBeInTheDocument();
    });

    it('renders the text input field', () => {
      renderWithTheme(darkTheme, <App />);
      expect(
        screen.getByPlaceholderText('Ask a question about your knowledge base...')
      ).toBeInTheDocument();
    });

    it('renders the send button disabled initially', () => {
      renderWithTheme(darkTheme, <App />);
      const sendButton = screen.getByRole('button', { name: /send/i });
      expect(sendButton).toBeDisabled();
    });

    it('renders the clear chat button', () => {
      renderWithTheme(darkTheme, <App />);
      expect(screen.getByRole('button', { name: /clear chat/i })).toBeInTheDocument();
    });
  });

  describe('Input interaction', () => {
    it('enables send button when input has text', () => {
      renderWithTheme(darkTheme, <App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');
      const sendButton = screen.getByRole('button', { name: /send/i });

      expect(sendButton).toBeDisabled();

      fireEvent.change(input, { target: { value: 'Hello world' } });
      expect(sendButton).toBeEnabled();
    });

    it('disables send button when input is only whitespace', () => {
      renderWithTheme(darkTheme, <App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');
      const sendButton = screen.getByRole('button', { name: /send/i });

      fireEvent.change(input, { target: { value: '   ' } });
      expect(sendButton).toBeDisabled();
    });
  });

  describe('Sending messages', () => {
    it('adds user message to chat on submit', async () => {
      mockAxiosPost.mockResolvedValueOnce({
        data: {
          answer: 'Test answer',
          sources: [],
          processing_time_seconds: 1.5,
        },
      });

      const { container } = renderWithTheme(darkTheme, <App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      submitForm(container);

      expect(screen.getByText('What is this?')).toBeInTheDocument();
    });

    it('clears input after sending message', async () => {
      mockAxiosPost.mockResolvedValueOnce({
        data: {
          answer: 'Test answer',
          sources: [],
          processing_time_seconds: 1.5,
        },
      });

      const { container } = renderWithTheme(darkTheme, <App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      submitForm(container);

      expect(input.value).toBe('');
    });

    it('shows loading state while waiting for response', async () => {
      mockAxiosPost.mockImplementation(
        () => new Promise((resolve) => setTimeout(resolve, 100))
      );

      const { container } = renderWithTheme(darkTheme, <App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      submitForm(container);

      expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });

    it('adds bot message with answer on successful response', async () => {
      mockAxiosPost.mockResolvedValueOnce({
        data: {
          answer: 'This is the answer',
          sources: [],
          processing_time_seconds: 2.3,
        },
      });

      const { container } = renderWithTheme(darkTheme, <App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      submitForm(container);

      await waitFor(() => {
        expect(screen.getByText('This is the answer')).toBeInTheDocument();
      });
    });

    it('disables send button while loading', async () => {
      mockAxiosPost.mockImplementation(
        () => new Promise((resolve) => setTimeout(resolve, 100))
      );

      const { container } = renderWithTheme(darkTheme, <App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      submitForm(container);

      const sendButton = screen.getByRole('button', { name: /thinking/i });
      expect(sendButton).toBeDisabled();
    });
  });

  describe('Sources display', () => {
    it('displays sources when returned from API', async () => {
      mockAxiosPost.mockResolvedValueOnce({
        data: {
          answer: 'Here is the answer',
          sources: [
            {
              title: 'Source Document 1',
              score: 0.95,
              document_id: 'abc12345-def6-7890',
              content_preview: 'This is a preview...',
            },
            {
              title: 'Source Document 2',
              score: 0.82,
              document_id: 'xyz98765-abc1-2345',
            },
          ],
          processing_time_seconds: 1.0,
        },
      });

      const { container } = renderWithTheme(darkTheme, <App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      submitForm(container);

      await waitFor(() => {
        expect(screen.getByText('Sources (2):')).toBeInTheDocument();
      });
      await waitFor(() => {
        expect(screen.getByText('Source Document 1')).toBeInTheDocument();
      });
      await waitFor(() => {
        expect(screen.getByText('Source Document 2')).toBeInTheDocument();
      });
      await waitFor(() => {
        expect(screen.getByText(/Relevance: 95%/)).toBeInTheDocument();
      });
      await waitFor(() => {
        expect(screen.getByText(/This is a preview/)).toBeInTheDocument();
      });
    });

    it('does not display sources section when no sources returned', async () => {
      mockAxiosPost.mockResolvedValueOnce({
        data: {
          answer: 'Here is the answer',
          sources: [],
          processing_time_seconds: 1.0,
        },
      });

      const { container } = renderWithTheme(darkTheme, <App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      submitForm(container);

      await waitFor(() => {
        expect(screen.getByText('Here is the answer')).toBeInTheDocument();
      });
      expect(screen.queryByText(/Sources/)).not.toBeInTheDocument();
    });
  });

  describe('Processing time display', () => {
    it('displays processing time when returned from API', async () => {
      mockAxiosPost.mockResolvedValueOnce({
        data: {
          answer: 'Here is the answer',
          sources: [],
          processing_time_seconds: 3.7,
        },
      });

      const { container } = renderWithTheme(darkTheme, <App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      submitForm(container);

      await waitFor(() => {
        expect(screen.getByText('(3.7s)')).toBeInTheDocument();
      });
    });
  });

  describe('Error handling', () => {
    it('shows error alert when API call fails', async () => {
      mockAxiosPost.mockRejectedValueOnce(new Error('Network error'));

      const { container } = renderWithTheme(darkTheme, <App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      submitForm(container);

      await waitFor(() => {
        expect(
          screen.getByText('Failed to get response. Please try again.')
        ).toBeInTheDocument();
      });
    });

    it('adds error bot message to chat on API failure', async () => {
      mockAxiosPost.mockRejectedValueOnce(new Error('Network error'));

      const { container } = renderWithTheme(darkTheme, <App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      submitForm(container);

      await waitFor(() => {
        expect(
          screen.getByText('Sorry, I encountered an error processing your question. Please try again.')
        ).toBeInTheDocument();
      });
    });
  });

  describe('Clear chat', () => {
    it('clears all messages when clear chat is clicked', async () => {
      mockAxiosPost.mockResolvedValueOnce({
        data: {
          answer: 'Test answer',
          sources: [],
          processing_time_seconds: 1.0,
        },
      });

      const { container } = renderWithTheme(darkTheme, <App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      submitForm(container);

      await waitFor(() => {
        expect(screen.getByText('Test answer')).toBeInTheDocument();
      });

      const clearButton = screen.getByRole('button', { name: /clear chat/i });
      fireEvent.click(clearButton);

      expect(screen.queryByText('What is this?')).not.toBeInTheDocument();
      expect(screen.queryByText('Test answer')).not.toBeInTheDocument();
      expect(
        screen.getByText('Start by asking a question about your personal knowledge base.')
      ).toBeInTheDocument();
    });

    it('clears error state when clear chat is clicked', async () => {
      mockAxiosPost.mockRejectedValueOnce(new Error('Network error'));

      const { container } = renderWithTheme(darkTheme, <App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      submitForm(container);

      await waitFor(() => {
        expect(
          screen.getByText('Failed to get response. Please try again.')
        ).toBeInTheDocument();
      });

      const clearButton = screen.getByRole('button', { name: /clear chat/i });
      fireEvent.click(clearButton);

      expect(
        screen.queryByText('Failed to get response. Please try again.')
      ).not.toBeInTheDocument();
    });
  });

  describe('Multiple messages', () => {
    it('maintains conversation history', async () => {
      mockAxiosPost
        .mockResolvedValueOnce({
          data: { answer: 'Answer 1', sources: [], processing_time_seconds: 1.0 },
        })
        .mockResolvedValueOnce({
          data: { answer: 'Answer 2', sources: [], processing_time_seconds: 2.0 },
        });

      const { container } = renderWithTheme(darkTheme, <App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'Question 1' } });
      submitForm(container);

      await waitFor(() => {
        expect(screen.getByText('Answer 1')).toBeInTheDocument();
      });

      fireEvent.change(input, { target: { value: 'Question 2' } });
      submitForm(container);

      await waitFor(() => {
        expect(screen.getByText('Question 1')).toBeInTheDocument();
      });
      await waitFor(() => {
        expect(screen.getByText('Answer 1')).toBeInTheDocument();
      });
      await waitFor(() => {
        expect(screen.getByText('Question 2')).toBeInTheDocument();
      });
      await waitFor(() => {
        expect(screen.getByText('Answer 2')).toBeInTheDocument();
      });
    });
  });
});
