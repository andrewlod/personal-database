import { createTheme } from '@mui/material/styles';

const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#388e3c',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
    messageBox: {
      user: {
        bg: '#e3f2fd',
        border: '#90caf9',
      },
      bot: {
        bg: '#f1f8e9',
        border: '#aed581',
      },
    },
  },
  components: {
    MuiListItem: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          border: '1px solid',
          marginBottom: 8,
          boxShadow: '0 1px 3px rgba(0,0,0,0.12)',
        },
      },
    },
  },
});

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#81c784',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    messageBox: {
      user: {
        bg: '#1a3a5c',
        border: '#1565c0',
      },
      bot: {
        bg: '#1b3a1b',
        border: '#388e3c',
      },
    },
  },
  components: {
    MuiListItem: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          border: '1px solid',
          marginBottom: 8,
          boxShadow: '0 1px 3px rgba(0,0,0,0.4)',
        },
      },
    },
  },
});

export { lightTheme, darkTheme };
