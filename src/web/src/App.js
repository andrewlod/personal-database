import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Container, Box, TextField, Button, Avatar, List, ListItem, ListItemAvatar, ListItemText, Typography, CircularProgress, Alert, IconButton, useTheme } from '@mui/material';
import { ChatBubbleOutline, Send, Clear, Brightness4, Brightness7 } from '@mui/icons-material';
import Markdown from 'react-markdown';
import { useThemeContext } from './ThemeContext';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const theme = useTheme();
  const { mode, toggleTheme } = useThemeContext();

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const question = input.trim();
    setInput('');
    setError(null);
    setLoading(true);

    const userMessage = {
      id: Date.now().toString(),
      text: question,
      sender: 'user',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);

    try {
      const response = await axios.post('/api/query/', {
        question: question,
        top_k: 5
      });

      const botMessage = {
        id: Date.now().toString() + 'b',
        text: response.data.answer,
        sender: 'bot',
        timestamp: new Date(),
        sources: response.data.sources || [],
        processingTime: response.data.processing_time_seconds
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (err) {
      setError('Failed to get response. Please try again.');
      console.error('Query error:', err);
      
      const errorMessage = {
        id: Date.now().toString() + 'e',
        text: 'Sorry, I encountered an error processing your question. Please try again.',
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  const getMessageBoxStyle = (sender) => {
    const colors = theme.palette.messageBox[sender];
    return {
      bgcolor: colors.bg,
      borderColor: colors.border,
    };
  };

  const sourceBoxBg = mode === 'dark' ? 'grey.900' : 'white';

  return (
    <Container maxWidth="mx-auto" sx={{ height: '100vh', display: 'flex', flexDirection: 'column', bgcolor: 'background.default' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', px: 3, py: 1, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Typography variant="h6" component="h1">
          Personal Database
        </Typography>
        <IconButton onClick={toggleTheme} color="inherit" aria-label="toggle theme">
          {mode === 'dark' ? <Brightness7 /> : <Brightness4 />}
        </IconButton>
      </Box>
      <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 3 }}>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        <List>
          {messages.map((message, index) => (
            <ListItem key={message.id} sx={{ mb: 2, ...getMessageBoxStyle(message.sender) }} data-testid={`message-${message.id}`}>
              <ListItemAvatar>
                <Avatar sx={{ bgcolor: message.sender === 'user' ? theme.palette.primary.main : theme.palette.secondary.main }}>
                  {message.sender === 'user' ? <ChatBubbleOutline fontSize="small" /> : <Send fontSize="small" />}
                </Avatar>
              </ListItemAvatar>
              <ListItemText sx={{ flexGrow: 1 }}>
                <Box sx={{ textAlign: message.sender === 'user' ? 'right' : 'left' }}>
                  {message.sender === 'user' ? (
                    <Box sx={{ textAlign: 'right' }}>
                      <Typography variant="body2" color="text.secondary">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </Typography>
                      <Typography variant="body1" fontWeight="medium" color="primary">
                        {message.text}
                      </Typography>
                    </Box>
                  ) : (
                    <>
                      <Box sx={{ textAlign: 'left', mb: 1 }}>
                        <Typography variant="body2" color="text.secondary">
                          {new Date(message.timestamp).toLocaleTimeString()} 
                          {message.processingTime && (
                            <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                              ({message.processingTime}s)
                            </Typography>
                          )}
                        </Typography>
                        <Markdown>{message.text}</Markdown>
                        {message.sources && message.sources.length > 0 && (
                          <Box sx={{ mt: 2, p: 1, bgcolor: mode === 'dark' ? 'grey.800' : 'grey.50', borderRadius: 1 }}>
                            <Typography variant="caption" color="text.secondary" fontWeight="medium">
                              Sources ({message.sources.length}):
                            </Typography>
                            {message.sources.map((source, sourceIndex) => (
                              <Box key={sourceIndex} sx={{ mb: 1, p: 0.5, bgcolor: sourceBoxBg, borderRadius: 0.5 }}>
                                <Typography variant="body2" color="text.primary" sx={{ fontWeight: 500 }}>
                                  {source.title || 'Untitled'}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                  Relevance: {Math.round(source.score * 100)}% 
                                  {source.document_id && ` | Doc: ${source.document_id.substring(0, 8)}`}
                                </Typography>
                                {source.content_preview && (
                                  <Box sx={{ mt: 0.5 }}>
                                    <Typography variant="body2" color="text.secondary">
                                      "{source.content_preview}"
                                    </Typography>
                                  </Box>
                                )}
                              </Box>
                            ))}
                          </Box>
                        )}
                      </Box>
                    </>
                  )}
                </Box>
              </ListItemText>
            </ListItem>
          ))}
          {loading && (
            <ListItem sx={{ justifyContent: 'center' }}>
              <CircularProgress size={24} />
            </ListItem>
          )}
          {messages.length === 0 && !loading && !error && (
            <ListItem sx={{ justifyContent: 'center', opacity: 0.5 }}>
              <Typography variant="body2">
                Start by asking a question about your personal knowledge base.
              </Typography>
            </ListItem>
          )}
        </List>
        <div ref={messagesEndRef} />
      </Box>
      
      <Box sx={{ px: 3, pb: 2, borderTop: '1px solid', borderColor: 'divider', backgroundColor: 'background.paper' }}>
        <form onSubmit={sendMessage} sx={{ display: 'flex', gap: 2 }}>
          <TextField
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your knowledge base..."
            fullWidth
            sx={{ flexGrow: 1 }}
            disabled={loading}
          />
          <Button 
            variant="contained" 
            color="primary" 
            type="submit" 
            disabled={loading || !input.trim()}
            sx={{ height: '56px' }}
          >
            {loading ? 'Thinking...' : 'Send'}
            {!loading && <Send fontSize="small" sx={{ ml: 1 }} />}
          </Button>
        </form>
        
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Button 
            variant="text" 
            size="small" 
            color="error" 
            onClick={clearChat}
          >
            Clear Chat
            <Clear fontSize="small" sx={{ ml: 0.5 }} />
          </Button>
        </Box>
      </Box>
    </Container>
  );
}

export default App;
