import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import axios from 'axios';
import App from '../App';

jest.mock('axios');
jest.mock('react-markdown', () => {
  return function MockMarkdown({ children }) {
    const r = require('react');
    return r.createElement('div', { 'data-testid': 'markdown' }, children);
  };
});

const mockAxiosPost = axios.post;

describe('App', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Initial render', () => {
    it('renders the empty state message', () => {
      render(<App />);
      expect(
        screen.getByText('Start by asking a question about your personal knowledge base.')
      ).toBeInTheDocument();
    });

    it('renders the text input field', () => {
      render(<App />);
      expect(
        screen.getByPlaceholderText('Ask a question about your knowledge base...')
      ).toBeInTheDocument();
    });

    it('renders the send button disabled initially', () => {
      render(<App />);
      const sendButton = screen.getByRole('button', { name: /send/i });
      expect(sendButton).toBeDisabled();
    });

    it('renders the clear chat button', () => {
      render(<App />);
      expect(screen.getByRole('button', { name: /clear chat/i })).toBeInTheDocument();
    });
  });

  describe('Input interaction', () => {
    it('enables send button when input has text', () => {
      render(<App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');
      const sendButton = screen.getByRole('button', { name: /send/i });

      expect(sendButton).toBeDisabled();

      fireEvent.change(input, { target: { value: 'Hello world' } });
      expect(sendButton).toBeEnabled();
    });

    it('disables send button when input is only whitespace', () => {
      render(<App />);
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

      render(<App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      fireEvent.submit(input.closest('form'));

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

      render(<App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      fireEvent.submit(input.closest('form'));

      expect(input.value).toBe('');
    });

    it('shows loading state while waiting for response', async () => {
      mockAxiosPost.mockImplementation(
        () => new Promise((resolve) => setTimeout(resolve, 100))
      );

      render(<App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      fireEvent.submit(input.closest('form'));

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

      render(<App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      fireEvent.submit(input.closest('form'));

      await waitFor(() => {
        expect(screen.getByText('This is the answer')).toBeInTheDocument();
      });
    });

    it('disables send button while loading', async () => {
      mockAxiosPost.mockImplementation(
        () => new Promise((resolve) => setTimeout(resolve, 100))
      );

      render(<App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      fireEvent.submit(input.closest('form'));

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

      render(<App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      fireEvent.submit(input.closest('form'));

      await waitFor(() => {
        expect(screen.getByText('Sources (2):')).toBeInTheDocument();
        expect(screen.getByText('Source Document 1')).toBeInTheDocument();
        expect(screen.getByText('Source Document 2')).toBeInTheDocument();
        expect(screen.getByText(/Relevance: 95%/)).toBeInTheDocument();
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

      render(<App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      fireEvent.submit(input.closest('form'));

      await waitFor(() => {
        expect(screen.getByText('Here is the answer')).toBeInTheDocument();
        expect(screen.queryByText(/Sources/)).not.toBeInTheDocument();
      });
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

      render(<App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      fireEvent.submit(input.closest('form'));

      await waitFor(() => {
        expect(screen.getByText('(3.7s)')).toBeInTheDocument();
      });
    });
  });

  describe('Error handling', () => {
    it('shows error alert when API call fails', async () => {
      mockAxiosPost.mockRejectedValueOnce(new Error('Network error'));

      render(<App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      fireEvent.submit(input.closest('form'));

      await waitFor(() => {
        expect(
          screen.getByText('Failed to get response. Please try again.')
        ).toBeInTheDocument();
      });
    });

    it('adds error bot message to chat on API failure', async () => {
      mockAxiosPost.mockRejectedValueOnce(new Error('Network error'));

      render(<App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      fireEvent.submit(input.closest('form'));

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

      render(<App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      fireEvent.submit(input.closest('form'));

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

      render(<App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'What is this?' } });
      fireEvent.submit(input.closest('form'));

      await waitFor(() => {
        expect(
          screen.getByText('Failed to get response. Please try again.')
        ).toBeInTheDocument();
      });

      const clearButton = screen.getByRole('button', { name: /clear chat/i });
      await act(async () => {
        fireEvent.click(clearButton);
      });

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

      render(<App />);
      const input = screen.getByPlaceholderText('Ask a question about your knowledge base...');

      fireEvent.change(input, { target: { value: 'Question 1' } });
      fireEvent.submit(input.closest('form'));

      await waitFor(() => {
        expect(screen.getByText('Answer 1')).toBeInTheDocument();
      });

      fireEvent.change(input, { target: { value: 'Question 2' } });
      fireEvent.submit(input.closest('form'));

      await waitFor(() => {
        expect(screen.getByText('Question 1')).toBeInTheDocument();
        expect(screen.getByText('Answer 1')).toBeInTheDocument();
        expect(screen.getByText('Question 2')).toBeInTheDocument();
        expect(screen.getByText('Answer 2')).toBeInTheDocument();
      });
    });
  });
});
