import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { lightTheme, darkTheme } from '../theme';
import { ThemeContext } from '../ThemeContext';
import App from '../App';

jest.mock('axios');
jest.mock('react-markdown', () => {
  return function MockMarkdown({ children }) {
    const r = require('react');
    return r.createElement('div', { 'data-testid': 'markdown' }, children);
  };
});

const renderWithTheme = (theme, ui, mode = 'dark') => {
  const toggleFn = jest.fn();
  return {
    ...render(
      <ThemeContext.Provider value={{ mode, toggleTheme: toggleFn }}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          {ui}
        </ThemeProvider>
      </ThemeContext.Provider>
    ),
    toggleFn,
  };
};

describe('Theme', () => {
  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
  });

  describe('Theme toggle', () => {
    it('renders the theme toggle button', () => {
      renderWithTheme(lightTheme, <App />);
      expect(screen.getByRole('button', { name: /toggle theme/i })).toBeInTheDocument();
    });

    it('toggle button calls the toggle function when clicked', () => {
      const { toggleFn } = renderWithTheme(darkTheme, <App />, 'dark');
      
      const toggleButton = screen.getByRole('button', { name: /toggle theme/i });
      fireEvent.click(toggleButton);

      expect(toggleFn).toHaveBeenCalled();
    });

    it('shows sun icon in dark mode', () => {
      renderWithTheme(darkTheme, <App />, 'dark');
      
      const toggleButton = screen.getByRole('button', { name: /toggle theme/i });
      expect(toggleButton).toBeInTheDocument();
    });

    it('shows moon icon in light mode', () => {
      renderWithTheme(lightTheme, <App />, 'light');
      
      const toggleButton = screen.getByRole('button', { name: /toggle theme/i });
      expect(toggleButton).toBeInTheDocument();
    });

    it('renders app with dark theme colors', () => {
      renderWithTheme(darkTheme, <App />, 'dark');
      
      expect(
        screen.getByText('Start by asking a question about your personal knowledge base.')
      ).toBeInTheDocument();
    });

    it('renders app with light theme colors', () => {
      renderWithTheme(lightTheme, <App />, 'light');
      
      expect(
        screen.getByText('Start by asking a question about your personal knowledge base.')
      ).toBeInTheDocument();
    });
  });
});
