import pandas as pd
import torch
import psutil
import threading
import time
import gc
import logging
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics.eval_statistical import (
    KolmogorovSmirnovTest, JensenShannonDistance, WassersteinDistance,
    MaximumMeanDiscrepancy, InverseKLDivergence
)
from sklearn.preprocessing import LabelEncoder

# Set up logging
logger = logging.getLogger(__name__)

class SynthCitySyntheticGenerator:
    """
    Modular SynthCity-based synthetic data generator with performance optimizations.
    """
    def __init__(self, algorithm="ddpm", target_column="Survived", plugin_kwargs=None):
        self.algorithm = algorithm
        self.target_column = target_column
        self.plugin_kwargs = plugin_kwargs or {}
        self.model = None
        self._monitoring = False
        
        # --- PERFORMANCE SAFEGUARDS ---
        logger.info(f"Initializing {algorithm} with performance optimizations")
        
        # Limit CPU usage for intensive models
        if algorithm in ['ddpm', 'ctgan', 'tvae', 'dpgan', 'pategan']:
            torch.set_num_threads(min(4, psutil.cpu_count() // 2))  # Use half of available CPUs
            logger.info(f"Limited CPU threads to {torch.get_num_threads()} for {algorithm}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory
                logger.info("Cleared GPU memory cache")

    def fit(self, real_data):
        """Fit the model with resource monitoring and memory optimization"""
        logger.info(f"Starting training for {self.algorithm} with {len(real_data)} samples")
        print("Test.......................1A")
        # Check system resources before training
        self._check_system_resources()
        print("Test.......................1B")
        # Start resource monitoring
        monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitoring = True
        monitor_thread.start()
        print("Test.......................1C")
        try:
            loader = GenericDataLoader(real_data, target=self.target_column)
            print("Test.......................1D")
            # Apply memory-efficient settings for intensive models
            if self.algorithm in ['ddpm', 'ctgan', 'tvae']:
                # Reduce batch size for memory-intensive models if not specified
                if 'batch_size' not in self.plugin_kwargs:
                    self.plugin_kwargs['batch_size'] = min(500, len(real_data) // 10)
                    logger.info(f"Set batch_size to {self.plugin_kwargs['batch_size']} for memory efficiency")
                
                # Add memory optimization for DDPM
                if self.algorithm == 'ddpm' and 'lr' not in self.plugin_kwargs:
                    self.plugin_kwargs['lr'] = 1e-3  # Conservative learning rate
            print("Test.......................1E")
            logger.info(f"Plugin kwargs: {self.plugin_kwargs}")
            self.model = Plugins().get(self.algorithm, **self.plugin_kwargs)
            print("Test.......................1F")
            self.model.fit(loader)
            print("Test.......................1G")
            logger.info(f"Training completed for {self.algorithm}")
            
        except Exception as e:
            logger.error(f"Training failed for {self.algorithm}: {str(e)}")
            raise
        finally:
            self._monitoring = False
            # Clean up resources
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return self

    def _fix_data_types(self, df):
        """Fix problematic data types before saving/returning"""
        logger.info("Fixing data types for compatibility")
        
        # Fix object columns that cause PyArrow issues
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Try to convert to string first
                    df[col] = df[col].astype(str)
                    # Replace any NaN strings
                    df[col] = df[col].replace(['nan', 'None', 'null'], '')
                except Exception as e:
                    logger.warning(f"Could not fix column {col}: {e}")
                    # As last resort, convert to category
                    df[col] = df[col].astype('category')
        
        # Ensure numeric columns are proper types
        for col in df.select_dtypes(include=['float64']).columns:
            if df[col].dtype == 'float64':
                # Check if it can be int
                if df[col].dropna().apply(lambda x: x.is_integer() if pd.notna(x) else True).all():
                    df[col] = df[col].astype('Int64')  # Nullable integer
        
        return df

    def sample(self, num_rows=1000):
        print("Test.......................2")
        """Generate synthetic samples with memory management and type fixing"""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        logger.info(f"Generating {num_rows} synthetic samples")
        
        try:
            # Generate data
            synth = self.model.generate(count=num_rows)
            print("Test.......................2A")
            result = synth.dataframe()
            print("Test.......................2B")
            logger.info(f"Raw generation completed: {result.shape}")
            
            # Fix data types immediately
            print("Test.......................2C")
            result = self._fix_data_types(result)
            logger.info("Data types fixed")
            
            # # Clean up memory immediately
            # del synth
            # gc.collect()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            print("Test.......................3")
            logger.info(f"Successfully generated {len(result)} synthetic samples")
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            # Attempt memory cleanup on failure
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            # Attempt memory cleanup on failure
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def generate(self, real_data, num_rows=1000):
        """Fit and generate in one call with comprehensive resource management"""
        print("Test.......................0")
        self.fit(real_data)
        print("Test.......................1")
        return self.sample(num_rows=num_rows)

    def _check_system_resources(self):
        """Check if system has enough resources for training"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        logger.info(f"System resources: CPU {cpu_percent}%, Memory {memory.percent}%")
        
        if memory.percent > 85:
            logger.warning(f"High memory usage ({memory.percent}%) - training may be slow")
        
        if cpu_percent > 90:
            logger.warning(f"High CPU usage ({cpu_percent}%) - training may be slow")
            
        # Check available disk space (for model checkpoints)
        disk = psutil.disk_usage('/')
        if (disk.free / disk.total) < 0.1:  # Less than 10% free space
            logger.warning(f"Low disk space ({(disk.free/disk.total)*100:.1f}% free)")

    def _monitor_resources(self):
        """Monitor CPU and memory usage during training"""
        logger.info("Starting resource monitoring")
        last_warning = 0
        
        while self._monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                # Only log warnings every 30 seconds to avoid spam
                current_time = time.time()
                if (cpu_percent > 95 or memory_percent > 90) and (current_time - last_warning > 30):
                    logger.warning(f"⚠️ High resource usage: CPU {cpu_percent}%, Memory {memory_percent}%")
                    last_warning = current_time
                    
                    # Force garbage collection on high usage
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {str(e)}")
                break
        
        logger.info("Resource monitoring stopped")

    @staticmethod
    def preprocess_for_evaluation(real_df, synth_df):
        """Preprocess data for evaluation with memory optimization"""
        logger.info("Preprocessing data for evaluation")
        
        real_df_processed = real_df.copy()
        synth_df_processed = synth_df.copy()
        cat_cols = real_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Encoding {len(cat_cols)} categorical columns: {cat_cols}")
        
        for col in cat_cols:
            try:
                le = LabelEncoder()
                all_values = pd.concat([real_df[[col]], synth_df[[col]]], axis=0).astype(str)
                le.fit(all_values.values.ravel())
                real_df_processed[col] = le.transform(real_df[col].astype(str))
                synth_df_processed[col] = le.transform(synth_df[col].astype(str))
            except Exception as e:
                logger.error(f"Failed to encode column {col}: {str(e)}")
                # Remove problematic columns
                real_df_processed = real_df_processed.drop(columns=[col])
                synth_df_processed = synth_df_processed.drop(columns=[col])
        
        return real_df_processed, synth_df_processed

    @staticmethod
    def evaluate_metrics_for_model(real_df, synth_df):
        """Evaluate model metrics with error handling"""
        logger.info("Evaluating model metrics")
        
        try:
            real_loader = GenericDataLoader(real_df, target_column="Survived")
            synth_loader = GenericDataLoader(synth_df, target_column="Survived")
        except Exception as e:
            logger.error(f"Failed to create data loaders: {str(e)}")
            return {metric: float('nan') for metric in ["KolmogorovSmirnov", "JensenShannon", "Wasserstein", "MaxMeanDiscrep", "InverseKLDiv"]}
        
        metrics = [
            ("KolmogorovSmirnov", KolmogorovSmirnovTest),
            ("JensenShannon", JensenShannonDistance),
            ("Wasserstein", WassersteinDistance),
            ("MaxMeanDiscrep", MaximumMeanDiscrepancy),
            ("InverseKLDiv", InverseKLDivergence)
        ]
        
        result = {}
        for name, metric_class in metrics:
            try:
                metric = metric_class()
                score = metric.evaluate(real_loader, synth_loader)
                if isinstance(score, dict):
                    val = sum(score.values()) / len(score) if score else float('nan')
                else:
                    val = score if score is not None else float('nan')
                result[name] = val
                logger.debug(f"Metric {name}: {val}")
            except Exception as e:
                logger.error(f"Failed to compute metric {name}: {str(e)}")
                result[name] = float('nan')
        
        logger.info(f"Metrics evaluation completed: {list(result.keys())}")
        return result