import time
import pandas as pd
import numpy as np
import logging
import gc
import psutil
from src.core.synth.synthcity_generator import SynthCitySyntheticGenerator
from src.core.synth.config import advanced_models, metric_cols

# Try to import torch for GPU memory management, fallback if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - GPU memory cleanup will be skipped")

logger = logging.getLogger(__name__)

def check_memory_before_model(model_name):
    """🛡️ ACTION ITEM 3: Memory circuit breaker - Check memory before training each model"""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        logger.info(f"🔍 Pre-{model_name} check: Memory {memory.percent:.1f}%, CPU {cpu_percent:.1f}%")
        
        if memory.percent > 85:
            logger.warning(f"⚠️ SKIPPING {model_name} - High memory usage ({memory.percent:.1f}%) detected")
            return False, f"Memory too high: {memory.percent:.1f}%"
        
        if memory.percent > 75:
            logger.warning(f"⚠️ {model_name} - Memory getting high ({memory.percent:.1f}%), proceeding with caution")
        
        return True, None
        
    except Exception as e:
        logger.error(f"Memory check failed for {model_name}: {e}")
        return True, None  # Default to proceed if check fails

def get_memory_usage_mb():
    """📊 ACTION ITEM 4: Detailed memory monitoring"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except:
        return 0

def aggressive_memory_cleanup(model_name):
    """🧹 ACTION ITEM 1: Force memory cleanup after each model"""
    try:
        # Force Python garbage collection
        collected = gc.collect()
        logger.info(f"🧹 Python GC collected {collected} objects after {model_name}")
        
        # Clear GPU memory if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info(f"🔧 GPU memory cleared after {model_name}")
        
        # Log memory status after cleanup
        memory = psutil.virtual_memory()
        process_memory = get_memory_usage_mb()
        logger.info(f"📊 After cleanup: System Memory {memory.percent:.1f}%, Process Memory {process_memory:.1f}MB")
        
    except Exception as e:
        logger.error(f"❌ Memory cleanup failed after {model_name}: {e}")

def log_memory_consumption(model_name, phase, start_memory_mb=None):
    """📈 ACTION ITEM 4: Monitor memory consumption per model"""
    try:
        current_memory = get_memory_usage_mb()
        system_memory = psutil.virtual_memory()
        
        log_msg = f"💾 {model_name} [{phase}]: Process {current_memory:.1f}MB, System {system_memory.percent:.1f}%"
        
        if start_memory_mb:
            memory_increase = current_memory - start_memory_mb
            log_msg += f", Delta +{memory_increase:.1f}MB"
            
            if memory_increase > 100:  # Alert if model uses >100MB
                logger.warning(f"⚠️ {model_name} high memory consumption: +{memory_increase:.1f}MB")
        
        logger.info(log_msg)
        return current_memory
        
    except Exception as e:
        logger.error(f"Memory logging failed for {model_name}: {e}")
        return 0

def synthcity_model_comparison(real_df, epochs=100, target_column="Survived"):
    """Enhanced model comparison with memory management and monitoring"""
    model_results = []
    total_models = len(advanced_models)
    
    # Initial system status
    logger.info(f"🚀 Starting evaluation of {total_models} models with {epochs} epochs")
    initial_memory = get_memory_usage_mb()
    logger.info(f"📊 Initial system state: Process Memory {initial_memory:.1f}MB")
    
    # Only DDPM reliably supports iteration parameters
    # ITERATION_PARAMS = {
    #     'ddpm': 'n_iter'
    # }
    ITERATION_PARAMS = {
        'ddpm': 'n_iter',
        'ctgan': 'n_iter',
        'tvae': 'n_iter', 
        'dpgan': 'n_iter', 
        'pategan': 'n_iter', 
        'privbayes': None,  # No iteration parameter
        'arf': None         # No iteration parameter
    }
    
    for idx, (model_name, plugin_name) in enumerate(advanced_models.items(), 1):
        model_start_time = time.time()
        logger.info(f"[{idx}/{total_models}] 🎯 Starting evaluation: {model_name}")
        
        # 🛡️ ACTION ITEM 3: Pre-model memory check
        should_proceed, skip_reason = check_memory_before_model(model_name)
        if not should_proceed:
            # Record as skipped
            skipped_record = {metric: None for metric in metric_cols}
            skipped_record.update({
                "Model": model_name,
                "Status": "Skipped",
                "Error": skip_reason,
                "Training_Time_Seconds": 0.0,
                "Epochs_Used": "N/A",
                "Sample_Size": 0
            })
            model_results.append(skipped_record)
            continue
        
        # 📈 ACTION ITEM 4: Track memory before model
        pre_model_memory = log_memory_consumption(model_name, "PRE-TRAINING")
        
        try:
            # Setup plugin parameters
            plugin_kwargs = {}
            iteration_param_name = ITERATION_PARAMS.get(plugin_name)
            if iteration_param_name:
                plugin_kwargs[iteration_param_name] = epochs
                logger.info(f"Using {iteration_param_name}={epochs} for {model_name}")
            else:
                logger.info(f"Using default parameters for {model_name}")

            # Create generator and train model
            generator = SynthCitySyntheticGenerator(
                algorithm=plugin_name,
                target_column=target_column,
                plugin_kwargs=plugin_kwargs
            )
            
            # 📈 Track memory during training
            log_memory_consumption(model_name, "DURING-INIT", pre_model_memory)
            
            # Generate data with conservative sample size
            sample_size = min(len(real_df), 1000)
            synth_data = generator.generate(real_data=real_df, num_rows=sample_size)
            
            # 📈 Track memory after generation
            post_generation_memory = log_memory_consumption(model_name, "POST-GENERATION", pre_model_memory)
            
            if synth_data is None or synth_data.empty:
                raise ValueError("Generated data is empty")
            
            # Evaluate model
            real_df_sample = real_df.sample(n=sample_size, random_state=42) if len(real_df) > sample_size else real_df
            real_df_enc, synth_df_enc = generator.preprocess_for_evaluation(real_df_sample, synth_data)
            metrics = generator.evaluate_metrics_for_model(real_df_enc, synth_df_enc)
            
            elapsed_time = time.time() - model_start_time
            
            # 📈 Track memory after evaluation
            log_memory_consumption(model_name, "POST-EVALUATION", pre_model_memory)
            
            # Create result record with safe values
            result_record = {}
            for metric_name, value in metrics.items():
                if pd.isna(value) or (isinstance(value, (int, float)) and not np.isfinite(value)):
                    result_record[metric_name] = None
                else:
                    result_record[metric_name] = float(value)
            
            result_record.update({
                "Model": model_name,
                "Status": "Success", 
                "Training_Time_Seconds": round(elapsed_time, 2),
                "Epochs_Used": epochs if iteration_param_name else "N/A",
                "Sample_Size": sample_size
            })
            
            logger.info(f"✅ {model_name} completed in {elapsed_time:.1f}s")
            model_results.append(result_record)
            
            # 🧹 ACTION ITEM 1: Explicit cleanup of large objects
            del generator
            del synth_data
            del real_df_enc
            del synth_df_enc
            del metrics
            
        except Exception as e:
            elapsed_time = time.time() - model_start_time
            error_msg = str(e)[:200]
            logger.error(f"❌ {model_name} failed after {elapsed_time:.1f}s: {error_msg}")
            
            # Create failed record
            failed_record = {metric: None for metric in metric_cols}
            failed_record.update({
                "Model": model_name,
                "Status": "Failed",
                "Error": error_msg,
                "Training_Time_Seconds": round(elapsed_time, 2),
                "Epochs_Used": epochs,
                "Sample_Size": 0
            })
            model_results.append(failed_record)
            
            # Clean up even on failure
            try:
                if 'generator' in locals():
                    del generator
                if 'synth_data' in locals():
                    del synth_data
            except:
                pass
        
        # 🧹 ACTION ITEM 1: Aggressive cleanup after each model
        aggressive_memory_cleanup(model_name)
        
        # 📈 ACTION ITEM 4: Log final memory state
        log_memory_consumption(model_name, "POST-CLEANUP", pre_model_memory)
        
        # Brief pause between models to allow memory settling
        time.sleep(1)

    # Final summary
    successful_models = [r for r in model_results if r["Status"] == "Success"]
    skipped_models = [r for r in model_results if r["Status"] == "Skipped"]
    failed_models = [r for r in model_results if r["Status"] == "Failed"]
    
    logger.info(f"📊 Evaluation Summary: {len(successful_models)} succeeded, "
                f"{len(failed_models)} failed, {len(skipped_models)} skipped")
    
    if skipped_models:
        skipped_names = [r["Model"] for r in skipped_models]
        logger.warning(f"⚠️ Skipped models due to memory constraints: {', '.join(skipped_names)}")
    
    # Final memory cleanup
    final_memory = get_memory_usage_mb()
    memory_growth = final_memory - initial_memory
    logger.info(f"📊 Final memory state: {final_memory:.1f}MB (growth: +{memory_growth:.1f}MB)")
    
    df_metrics = pd.DataFrame(model_results)
    return df_metrics


def select_best_model(df_metrics, metric_cols):
    """Enhanced model selection with better status handling"""
    # Only consider successful runs
    successful_runs = df_metrics[df_metrics['Status'] == 'Success'].copy()
    
    if successful_runs.empty:
        # Check what happened
        skipped_count = len(df_metrics[df_metrics['Status'] == 'Skipped'])
        failed_count = len(df_metrics[df_metrics['Status'] == 'Failed'])
        
        if skipped_count > 0:
            logger.error(f"No models completed successfully. {skipped_count} skipped due to memory, {failed_count} failed")
            return f"No models completed (memory constraints)", None
        else:
            logger.error(f"No models completed successfully. {failed_count} failed")
            return "No models completed successfully", None
    
    logger.info(f"Selecting best from {len(successful_runs)} successful models")
    
    # Calculate aggregate score with pandas compatibility
    valid_metrics = []
    for _, row in successful_runs.iterrows():
        row_metrics = []
        for col in metric_cols:
            value = row[col]
            if value is not None and pd.notna(value) and (isinstance(value, (int, float)) and np.isfinite(value)):
                row_metrics.append(float(value))
        
        if row_metrics:
            valid_metrics.append(sum(row_metrics) / len(row_metrics))
        else:
            valid_metrics.append(float('inf'))
    
    if not any(score != float('inf') for score in valid_metrics):
        return "No valid metrics available", None
    
    successful_runs['AggregateScore'] = valid_metrics
    best_idx = successful_runs['AggregateScore'].idxmin()
    best_row = successful_runs.loc[best_idx]
    
    logger.info(f"🏆 Best model: {best_row['Model']} with score: {best_row['AggregateScore']:.4f}")
    
    return best_row['Model'], best_row