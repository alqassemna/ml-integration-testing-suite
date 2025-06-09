import React, { useState, useEffect } from 'react';
import { Play, Code, Database, Zap, CheckCircle, XCircle, Clock, BarChart3, Settings, Download } from 'lucide-react';

const MLIntegrationTestingSuite = () => {
  const [activeFramework, setActiveFramework] = useState('tensorflow');
  const [runningTests, setRunningTests] = useState({});
  const [testResults, setTestResults] = useState({});
  const [selectedTest, setSelectedTest] = useState('attention_comparison');

  const frameworks = {
    tensorflow: {
      name: 'TensorFlow 2.x',
      color: 'bg-orange-500',
      version: '2.15.0'
    },
    pytorch: {
      name: 'PyTorch',
      color: 'bg-red-500', 
      version: '2.1.0'
    }
  };

  const testSuites = {
    attention_comparison: {
      name: 'Attention Mechanisms Comparison',
      description: 'Benchmarks different attention mechanisms for TTS quality and performance',
      frameworks: ['tensorflow', 'pytorch'],
      duration: '15-30 min',
      tests: [
        'Content-Based Attention Performance',
        'Location-Aware Attention Quality',
        'Location-Based Attention Speed',
        'Decoder-Only Attention Memory Usage'
      ]
    },
    model_compatibility: {
      name: 'N8N Model Compatibility',
      description: 'Tests integration between ML models and N8N workflow execution',
      frameworks: ['tensorflow', 'pytorch'],
      duration: '10-20 min',
      tests: [
        'Model Loading in N8N Context',
        'Batch Processing Integration',
        'Real-time Inference Testing',
        'Error Handling Validation'
      ]
    },
    performance_benchmarks: {
      name: 'Performance Benchmarks',
      description: 'Comprehensive performance testing across different hardware configurations',
      frameworks: ['tensorflow', 'pytorch'],
      duration: '30-45 min',
      tests: [
        'CPU vs GPU Performance',
        'Memory Usage Profiling',
        'Latency Measurements',
        'Throughput Analysis'
      ]
    },
    quality_metrics: {
      name: 'Audio Quality Metrics',
      description: 'Automated quality assessment using MOS, WER, and other metrics',
      frameworks: ['tensorflow', 'pytorch'],
      duration: '20-35 min',
      tests: [
        'Mean Opinion Score (MOS)',
        'Word Error Rate (WER)',
        'Signal-to-Noise Ratio (SNR)',
        'Perceptual Quality Assessment'
      ]
    }
  };

  const codeExamples = {
    tensorflow: {
      attention_comparison: `# TensorFlow Attention Mechanisms Implementation
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class ContentBasedAttention(layers.Layer):
    def __init__(self, units):
        super(ContentBasedAttention, self).__init__()
        self.W_a = layers.Dense(units)
        self.U_a = layers.Dense(units)
        self.v_a = layers.Dense(1)
        
    def call(self, encoder_outputs, decoder_state):
        # Score calculation
        score = tf.nn.tanh(
            self.W_a(encoder_outputs) + 
            tf.expand_dims(self.U_a(decoder_state), 1)
        )
        attention_weights = tf.nn.softmax(self.v_a(score), axis=1)
        context_vector = attention_weights * encoder_outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class LocationAwareAttention(layers.Layer):
    def __init__(self, units, filters=32, kernel_size=31):
        super(LocationAwareAttention, self).__init__()
        self.W_a = layers.Dense(units)
        self.U_a = layers.Dense(units)
        self.v_a = layers.Dense(1)
        self.location_conv = layers.Conv1D(filters, kernel_size, 
                                         padding='same')
        self.location_dense = layers.Dense(units)
        
    def call(self, encoder_outputs, decoder_state, prev_attention):
        # Location feature computation
        location_features = self.location_conv(
            tf.expand_dims(prev_attention, -1)
        )
        location_features = self.location_dense(location_features)
        
        # Combined attention score
        score = tf.nn.tanh(
            self.W_a(encoder_outputs) + 
            tf.expand_dims(self.U_a(decoder_state), 1) +
            location_features
        )
        attention_weights = tf.nn.softmax(self.v_a(score), axis=1)
        context_vector = tf.reduce_sum(
            attention_weights * encoder_outputs, axis=1
        )
        return context_vector, attention_weights

# TTS Model with Configurable Attention
class ConfigurableTTSModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoder_units, 
                 decoder_units, attention_type='content_based'):
        super(ConfigurableTTSModel, self).__init__()
        
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.encoder = layers.LSTM(encoder_units, return_sequences=True)
        self.decoder = layers.LSTM(decoder_units, return_state=True)
        
        if attention_type == 'content_based':
            self.attention = ContentBasedAttention(decoder_units)
        elif attention_type == 'location_aware':
            self.attention = LocationAwareAttention(decoder_units)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
            
        self.mel_projection = layers.Dense(80)  # 80-dim mel spectrogram
        
    def call(self, inputs, training=None):
        # Encode input sequence
        embedded = self.embedding(inputs)
        encoder_outputs = self.encoder(embedded)
        
        # Initialize decoder state
        batch_size = tf.shape(inputs)[0]
        decoder_state = tf.zeros([batch_size, self.decoder.units])
        
        # Attention-based decoding (simplified)
        context, attention_weights = self.attention(
            encoder_outputs, decoder_state
        )
        
        # Generate mel spectrogram
        mel_output = self.mel_projection(context)
        
        return mel_output, attention_weights

# N8N Integration Test Function
def test_n8n_integration():
    """Test TensorFlow model integration with N8N workflows"""
    
    # Initialize model
    model = ConfigurableTTSModel(
        vocab_size=1000,
        embedding_dim=128,
        encoder_units=256,
        decoder_units=256,
        attention_type='content_based'
    )
    
    # Simulate N8N input format
    n8n_input = {
        'text': 'Neural statistical parametric speech synthesis',
        'voice_config': {
            'attention_type': 'content_based',
            'quality': 'high'
        }
    }
    
    # Process through model (simplified)
    text_tokens = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)
    mel_output, attention_weights = model(text_tokens)
    
    # Return N8N compatible output
    return {
        'audio_features': mel_output.numpy().tolist(),
        'attention_alignment': attention_weights.numpy().tolist(),
        'status': 'success',
        'processing_time': 0.45  # seconds
    }

# Performance Benchmarking
class PerformanceBenchmark:
    def __init__(self):
        self.results = {}
        
    def benchmark_attention_mechanisms(self):
        """Benchmark different attention mechanisms"""
        mechanisms = ['content_based', 'location_aware']
        
        for mechanism in mechanisms:
            model = ConfigurableTTSModel(
                vocab_size=1000,
                embedding_dim=128,
                encoder_units=256,
                decoder_units=256,
                attention_type=mechanism
            )
            
            # Warm up
            dummy_input = tf.random.uniform([1, 50], maxval=1000, dtype=tf.int32)
            model(dummy_input)
            
            # Benchmark
            start_time = tf.timestamp()
            for _ in range(100):
                _ = model(dummy_input)
            end_time = tf.timestamp()
            
            avg_time = (end_time - start_time) / 100
            
            self.results[mechanism] = {
                'avg_inference_time': float(avg_time),
                'memory_usage': self.get_memory_usage(),
                'quality_score': self.simulate_quality_score()
            }
            
        return self.results
    
    def get_memory_usage(self):
        """Simulate memory usage measurement"""
        return np.random.uniform(800, 1500)  # MB
    
    def simulate_quality_score(self):
        """Simulate quality score (MOS)"""
        return np.random.uniform(3.5, 4.8)`,
      
      model_compatibility: `# N8N Model Compatibility Testing
import tensorflow as tf
import json
import time
from typing import Dict, Any

class N8NTensorFlowAdapter:
    """Adapter class for TensorFlow models in N8N workflows"""
    
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self.preprocessing_pipeline = self.setup_preprocessing()
        
    def setup_preprocessing(self):
        """Setup text preprocessing pipeline"""
        return tf.keras.utils.get_custom_objects()
    
    def process_n8n_request(self, n8n_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request from N8N workflow"""
        try:
            # Extract and validate input
            text = n8n_data.get('text', '')
            config = n8n_data.get('config', {})
            
            if not text:
                raise ValueError("Empty text input")
            
            # Preprocess text
            processed_input = self.preprocess_text(text)
            
            # Run inference
            start_time = time.time()
            output = self.model(processed_input)
            inference_time = time.time() - start_time
            
            # Format output for N8N
            return {
                'status': 'success',
                'audio_features': output.numpy().tolist(),
                'metadata': {
                    'inference_time': inference_time,
                    'input_length': len(text),
                    'model_version': '1.0.0'
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'error_type': type(e).__name__
            }
    
    def preprocess_text(self, text: str):
        """Convert text to model input format"""
        # Simplified tokenization
        tokens = text.lower().split()
        token_ids = [hash(token) % 1000 for token in tokens]
        return tf.constant([token_ids], dtype=tf.int32)

# Batch Processing Test
def test_batch_processing():
    """Test batch processing capabilities"""
    
    # Create test batch
    batch_data = [
        {'text': 'Hello world', 'config': {'quality': 'high'}},
        {'text': 'Neural networks are amazing', 'config': {'quality': 'medium'}},
        {'text': 'TensorFlow integration testing', 'config': {'quality': 'high'}}
    ]
    
    adapter = N8NTensorFlowAdapter('path/to/model')
    results = []
    
    start_time = time.time()
    for item in batch_data:
        result = adapter.process_n8n_request(item)
        results.append(result)
    
    total_time = time.time() - start_time
    
    return {
        'batch_size': len(batch_data),
        'total_processing_time': total_time,
        'average_time_per_item': total_time / len(batch_data),
        'success_rate': sum(1 for r in results if r['status'] == 'success') / len(results),
        'results': results
    }`
    },
    pytorch: {
      attention_comparison: `# PyTorch Attention Mechanisms Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ContentBasedAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(ContentBasedAttention, self).__init__()
        self.encoder_projection = nn.Linear(encoder_dim, attention_dim)
        self.decoder_projection = nn.Linear(decoder_dim, attention_dim)
        self.attention_projection = nn.Linear(attention_dim, 1)
        
    def forward(self, encoder_outputs, decoder_state):
        # encoder_outputs: [batch_size, seq_len, encoder_dim]
        # decoder_state: [batch_size, decoder_dim]
        
        batch_size, seq_len, _ = encoder_outputs.size()
        
        # Project encoder outputs
        encoder_proj = self.encoder_projection(encoder_outputs)
        
        # Project and expand decoder state
        decoder_proj = self.decoder_projection(decoder_state)
        decoder_proj = decoder_proj.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # Compute attention scores
        attention_scores = torch.tanh(encoder_proj + decoder_proj)
        attention_scores = self.attention_projection(attention_scores).squeeze(-1)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Compute context vector
        context = torch.sum(attention_weights.unsqueeze(-1) * encoder_outputs, dim=1)
        
        return context, attention_weights

class LocationAwareAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim, 
                 location_filters=32, location_kernel=31):
        super(LocationAwareAttention, self).__init__()
        
        self.encoder_projection = nn.Linear(encoder_dim, attention_dim)
        self.decoder_projection = nn.Linear(decoder_dim, attention_dim)
        self.location_conv = nn.Conv1d(1, location_filters, 
                                     location_kernel, padding=location_kernel//2)
        self.location_projection = nn.Linear(location_filters, attention_dim)
        self.attention_projection = nn.Linear(attention_dim, 1)
        
    def forward(self, encoder_outputs, decoder_state, prev_attention):
        batch_size, seq_len, _ = encoder_outputs.size()
        
        # Content-based features
        encoder_proj = self.encoder_projection(encoder_outputs)
        decoder_proj = self.decoder_projection(decoder_state)
        decoder_proj = decoder_proj.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # Location-based features
        prev_attention = prev_attention.unsqueeze(1)  # [batch, 1, seq_len]
        location_features = self.location_conv(prev_attention)
        location_features = location_features.transpose(1, 2)  # [batch, seq_len, filters]
        location_proj = self.location_projection(location_features)
        
        # Combined attention
        attention_scores = torch.tanh(encoder_proj + decoder_proj + location_proj)
        attention_scores = self.attention_projection(attention_scores).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        context = torch.sum(attention_weights.unsqueeze(-1) * encoder_outputs, dim=1)
        
        return context, attention_weights

class ConfigurableTTSModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoder_dim, decoder_dim, 
                 mel_dim=80, attention_type='content_based'):
        super(ConfigurableTTSModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, encoder_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(mel_dim, decoder_dim, batch_first=True)
        
        if attention_type == 'content