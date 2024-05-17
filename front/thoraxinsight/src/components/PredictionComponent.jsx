import React, { useState } from 'react';
import axios from 'axios';
import { Box, Button, Container, Typography, CircularProgress, Avatar, Grid, TextField } from '@mui/material';
import CssBaseline from '@mui/material/CssBaseline';
import { styled, ThemeProvider } from '@mui/system';
import LocationSearchingIcon from '@mui/icons-material/LocationSearching';
import theme from '../styles/theme'; // Make sure to import the same theme
import DeleteIcon from '@mui/icons-material/Delete';
import '../styles/PredictionComponent.css';

const StyledInput = styled('input')({
  display: 'none',
});

const PredictionComponent = () => {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [taskname, setTaskname] = useState('');
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setPrediction(null); // Clear the previous prediction when a new file is selected
  };

  const handlePredict = async () => {
    const formData = new FormData();
    formData.append('file', file);
    const token = localStorage.getItem('token');
    const email = localStorage.getItem('email');
    setLoading(true);
  
    try {
      // Upload file using Fetch API
      const uploadResponse = await fetch('http://localhost:8000/files/uploadfile', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData, // Pass formData directly as the body
      });
  
      if (!uploadResponse.ok) {
        throw new Error('Failed to upload file');
      }
  
      const uploadData = await uploadResponse.json();
      const uploadDataPath = await `uploads/${uploadData.filename}`;
      console.log(uploadDataPath);
  
      // Now send the required data to another endpoint
      const taskResponse = await fetch(`http://localhost:8000/tasks/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body:JSON.stringify({
          name: taskname,
          user_email: email,
          input_path: uploadDataPath,
        }),
      });
  
      if (!taskResponse.ok) {
        throw new Error('Failed to create task');
      }
  
      const taskData = await taskResponse.json();
      console.log(taskData.prediction);
      setPrediction(taskData.prediction);
    } catch (error) {
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };
  
  

  const handleReset = () => {
    setFile(null);
    setPrediction(null);
  };

  return (
    <ThemeProvider theme={theme}>
      <Grid container component="main" sx={{ height: '100vh' }}>
        <CssBaseline />
        <Container maxWidth="sm" sx={{ mt: 4 }}>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              p: 3,
              border: '1px solid #ddd',
              borderRadius: '8px',
              boxShadow: 2,
              backgroundColor: 'background.default'
            }}
          >
            <Avatar sx={{ m: 1, bgcolor: 'text.secondary' }}>
              <LocationSearchingIcon />
            </Avatar>
            <Typography variant="h4" gutterBottom>
              Classify Image
            </Typography>
            <label htmlFor="upload-file">
              <StyledInput id="upload-file" type="file" onChange={handleFileChange} />
              <Button variant="contained" component="span" sx={{ mt: 2, mb: 2, bgcolor: 'success.main', '&:hover': { bgcolor: 'secondary.main' } }}>
                Select File
              </Button>
            </label>
            {file && (
              <Box sx={{ mt: 2, textAlign: 'center' }}>
                <Typography variant="body1">Archivo seleccionado: {file.name}</Typography>
                <img
                  src={URL.createObjectURL(file)}
                  alt="Preview"
                  style={{ marginTop: '10px', maxWidth: '100%', maxHeight: '200px' }}
                />
              </Box>
            )}
            <TextField
              id="taskname"
              label="Task Name"
              variant="outlined"
              margin="normal"
              value={taskname}
              onChange={(e) => setTaskname(e.target.value)}
            />
            <Button
              variant="contained"
              color="text"
              onClick={handlePredict}
              sx={{ mt: 3, bgcolor: 'success.main', '&:hover': { bgcolor: 'secondary.main' } }}
              disabled={!file || loading || !taskname}
            >
              {loading ? <CircularProgress size={24} /> : 'Predict'}
            </Button>
            <Button
              variant="outlined"
              color="text"
              onClick={handleReset}
              sx={{ mt: 2, color: 'text.primary', '&:hover': { bgcolor: 'secondary.main'}}}
              disabled={!file || loading}
              startIcon= {<DeleteIcon />}
            >
              Delete
            </Button>
            {prediction && (
              <Box sx={{ mt: 4, textAlign: 'center' }}>
                <Typography variant="h5">Predicción:</Typography>
                <Typography variant="body1">{prediction}</Typography>
              </Box>
            )}
          </Box>
        </Container>
      </Grid>
    </ThemeProvider>
  );
};

export default PredictionComponent;
