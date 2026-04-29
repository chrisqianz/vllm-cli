#!/usr/bin/env python3
"""
Validation schema builder for vLLM CLI configuration.

This module provides a declarative way to define validation rules
for vLLM configuration parameters, building on the validators module
to create a comprehensive validation system.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import CompositeValidator, DependencyValidator
from .factory import (
    create_boolean_validator,
    create_choice_validator,
    create_float_validator,
    create_integer_validator,
    create_string_validator,
    validate_non_negative_integer,
    validate_port_number,
    validate_positive_integer,
    validate_probability,
)
from .registry import ValidationRegistry

logger = logging.getLogger(__name__)

# Cached schema to avoid repeated file reads
_argument_schema_cache: Optional[Dict[str, Any]] = None


def _load_argument_schema() -> Dict[str, Any]:
    """Load argument_schema.json from the schemas package directory."""
    global _argument_schema_cache
    if _argument_schema_cache is not None:
        return _argument_schema_cache

    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "argument_schema.json"
    try:
        with open(schema_path, "r") as f:
            _argument_schema_cache = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load argument_schema.json: {e}")
        _argument_schema_cache = {}
    return _argument_schema_cache


def _get_choice_values(arg_name: str) -> List[str]:
    """Get choice values from argument_schema.json, filtering out None."""
    schema = _load_argument_schema()
    arg_info = schema.get("arguments", {}).get(arg_name, {})
    if arg_info.get("type") == "choice":
        return [c for c in arg_info.get("choices", []) if c is not None]
    return []


def create_vllm_validation_registry() -> ValidationRegistry:
    """
    Create a validation registry for vLLM configuration parameters.

    This function builds a comprehensive validation registry based on
    the vLLM argument schema, providing type checking, range validation,
    and dependency checking for all supported parameters.

    Returns:
        ValidationRegistry configured for vLLM parameters
    """
    registry = ValidationRegistry()

    # Core server parameters
    registry.register("port", validate_port_number("port"))

    registry.register("host", create_string_validator("host", min_length=1))

    # Model parameters (choices loaded from argument_schema.json)
    registry.register(
        "dtype",
        create_choice_validator(
            "dtype",
            _get_choice_values("dtype") or ["auto", "float16", "bfloat16", "float32", "float8"],
        ),
    )

    registry.register(
        "max_model_len",
        create_integer_validator("max_model_len", min_value=1, required=False),
    )

    registry.register("max_num_seqs", validate_positive_integer("max_num_seqs"))

    # Parallelism parameters
    registry.register(
        "tensor_parallel_size", validate_positive_integer("tensor_parallel_size")
    )

    registry.register(
        "pipeline_parallel_size", validate_positive_integer("pipeline_parallel_size")
    )

    # Memory management
    registry.register(
        "gpu_memory_utilization", validate_probability("gpu_memory_utilization")
    )

    registry.register(
        "max_num_batched_tokens", validate_positive_integer("max_num_batched_tokens")
    )

    registry.register("max_paddings", validate_non_negative_integer("max_paddings"))

    # Quantization parameters (choices loaded from argument_schema.json)
    registry.register(
        "quantization",
        create_choice_validator(
            "quantization",
            _get_choice_values("quantization")
            or [
                "awq",
                "awq_marlin",
                "auto-round",
                "bitsandbytes",
                "bitblas",
                "gguf",
                "gptq",
                "gptq_marlin",
                "fp8",
                "fbgemm_fp8",
                "compressed-tensors",
                "experts_int8",
                "modelopt_fp4",
                "modelopt_mxfp8",
            ],
        ),
    )

    # KV Cache parameters (choices loaded from argument_schema.json)
    registry.register(
        "kv_cache_dtype",
        create_choice_validator(
            "kv_cache_dtype",
            _get_choice_values("kv_cache_dtype") or ["auto", "fp8", "fp16", "bf16"],
        ),
    )

    registry.register(
        "block_size",
        create_choice_validator(
            "block_size",
            _get_choice_values("block_size") or [8, 16, 32, 64, 128],
        ),
    )

    # Attention parameters
    registry.register(
        "enable_prefix_caching", create_boolean_validator("enable_prefix_caching")
    )

    registry.register(
        "enable_chunked_prefill", create_boolean_validator("enable_chunked_prefill")
    )

    registry.register(
        "max_num_on_the_fly_seq_groups",
        validate_positive_integer("max_num_on_the_fly_seq_groups"),
    )

    # Performance parameters
    registry.register("enforce_eager", create_boolean_validator("enforce_eager"))

    registry.register(
        "max_context_len_to_capture",
        validate_positive_integer("max_context_len_to_capture"),
    )

    # Loading parameters
    registry.register(
        "load_format",
        create_choice_validator(
            "load_format", ["auto", "pt", "safetensors", "npcache", "dummy"]
        ),
    )

    registry.register(
        "download_dir", create_string_validator("download_dir", min_length=1)
    )

    # CPU offloading
    registry.register(
        "cpu_offload_gb", create_float_validator("cpu_offload_gb", min_value=0.0)
    )

    # Trust and safety parameters
    registry.register(
        "trust_remote_code", create_boolean_validator("trust_remote_code")
    )

    registry.register(
        "disable_log_stats", create_boolean_validator("disable_log_stats")
    )

    registry.register(
        "disable_log_requests", create_boolean_validator("disable_log_requests")
    )

    # Advanced parameters
    registry.register("revision", create_string_validator("revision", min_length=1))

    registry.register(
        "tokenizer_revision",
        create_string_validator("tokenizer_revision", min_length=1),
    )

    registry.register("seed", create_integer_validator("seed", min_value=0))

    # Worker and distribution parameters
    registry.register("worker_use_ray", create_boolean_validator("worker_use_ray"))

    registry.register(
        "ray_workers_use_nsight", create_boolean_validator("ray_workers_use_nsight")
    )

    # Speculative decoding
    registry.register(
        "speculative_model", create_string_validator("speculative_model", min_length=1)
    )

    registry.register(
        "num_speculative_tokens", validate_positive_integer("num_speculative_tokens")
    )

    # LoRA parameters
    registry.register("enable_lora", create_boolean_validator("enable_lora"))

    registry.register("max_loras", validate_positive_integer("max_loras"))

    registry.register("max_lora_rank", validate_positive_integer("max_lora_rank"))

    registry.register(
        "lora_extra_vocab_size", validate_non_negative_integer("lora_extra_vocab_size")
    )

    # Add dependency validators
    _add_dependency_validators(registry)

    return registry


def _add_dependency_validators(registry: ValidationRegistry) -> None:
    """
    Add dependency validators for parameters that require other parameters.

    Args:
        registry: ValidationRegistry to add dependencies to
    """
    # LoRA dependencies
    lora_dependent_params = ["max_loras", "max_lora_rank", "lora_extra_vocab_size"]
    for param in lora_dependent_params:
        validator = registry.get_validator(param)
        if validator:
            validator.add_validator(DependencyValidator(param, ["enable_lora"]))

    # Speculative decoding dependencies
    spec_dependent_params = ["num_speculative_tokens"]
    for param in spec_dependent_params:
        validator = registry.get_validator(param)
        if validator:
            validator.add_validator(DependencyValidator(param, ["speculative_model"]))

    # Ray dependencies
    ray_dependent_params = ["ray_workers_use_nsight"]
    for param in ray_dependent_params:
        validator = registry.get_validator(param)
        if validator:
            validator.add_validator(DependencyValidator(param, ["worker_use_ray"]))


def create_compatibility_validator(
    registry: ValidationRegistry,
) -> "CompatibilityValidator":
    """
    Create a compatibility validator for checking parameter combinations.

    Args:
        registry: Base validation registry

    Returns:
        CompatibilityValidator for checking parameter interactions
    """
    return CompatibilityValidator(registry)


class CompatibilityValidator:
    """
    Validator for checking compatibility between configuration parameters.

    This validator checks for known incompatible parameter combinations
    and provides warnings or errors for potentially problematic configurations.
    """

    def __init__(self, base_registry: ValidationRegistry):
        self.base_registry = base_registry
        self.compatibility_rules = self._build_compatibility_rules()

    def _build_compatibility_rules(self) -> List[Dict[str, Any]]:
        """Build list of compatibility rules."""
        return [
            {
                "name": "eager_mode_conflicts",
                "condition": lambda config: (
                    config.get("enforce_eager") and config.get("enable_prefix_caching")
                ),
                "message": "enforce_eager disables CUDA graphs, which may conflict with prefix caching",
                "severity": "warning",
            },
            {
                "name": "multiple_parallelism",
                "condition": lambda config: (
                    config.get("tensor_parallel_size", 1) > 1
                    and config.get("pipeline_parallel_size", 1) > 1
                ),
                "message": "Using both tensor and pipeline parallelism may not be optimal",
                "severity": "warning",
            },
            {
                "name": "cpu_offload_high_gpu_util",
                "condition": lambda config: (
                    config.get("cpu_offload_gb", 0) > 0
                    and config.get("gpu_memory_utilization", 0.9) > 0.9
                ),
                "message": "CPU offloading with high GPU utilization may cause memory thrashing",
                "severity": "warning",
            },
            {
                "name": "high_parallelism_small_model",
                "condition": lambda config: (
                    config.get("tensor_parallel_size", 1) > 4
                    and "max_model_len" in config
                    and config["max_model_len"] < 8192
                ),
                "message": "High tensor parallelism with small context may be inefficient",
                "severity": "warning",
            },
        ]

    def validate_compatibility(self, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Check configuration for compatibility issues.

        Args:
            config: Configuration dictionary to check

        Returns:
            List of compatibility issues with severity levels
        """
        issues = []

        for rule in self.compatibility_rules:
            try:
                if rule["condition"](config):
                    issues.append(
                        {
                            "rule": rule["name"],
                            "message": rule["message"],
                            "severity": rule["severity"],
                        }
                    )
            except Exception as e:
                logger.warning(f"Error checking compatibility rule {rule['name']}: {e}")

        return issues


def load_validation_schema_from_file(schema_file: Path) -> ValidationRegistry:
    """
    Load validation schema from a JSON schema file.

    This allows for external configuration of validation rules
    without modifying the code.

    Args:
        schema_file: Path to JSON schema file

    Returns:
        ValidationRegistry built from the schema file
    """
    registry = ValidationRegistry()

    try:
        with open(schema_file, "r") as f:
            schema_data = json.load(f)

        arguments = schema_data.get("arguments", {})

        for arg_name, arg_info in arguments.items():
            validator = _create_validator_from_schema(arg_name, arg_info)
            if validator:
                registry.register(arg_name, validator)

        logger.info(f"Loaded validation schema from {schema_file}")

    except Exception as e:
        logger.error(f"Failed to load validation schema from {schema_file}: {e}")
        # Fall back to default schema
        registry = create_vllm_validation_registry()

    return registry


def _create_validator_from_schema(
    arg_name: str, arg_info: Dict[str, Any]
) -> Optional[CompositeValidator]:
    """
    Create a validator from schema information.

    Args:
        arg_name: Argument name
        arg_info: Argument information from schema

    Returns:
        CompositeValidator or None if creation fails
    """
    try:
        arg_type = arg_info.get("type")
        validation_info = arg_info.get("validation", {})

        if arg_type == "integer":
            return create_integer_validator(
                arg_name,
                min_value=validation_info.get("min"),
                max_value=validation_info.get("max"),
            )
        elif arg_type == "float":
            return create_float_validator(
                arg_name,
                min_value=validation_info.get("min"),
                max_value=validation_info.get("max"),
            )
        elif arg_type == "string":
            return create_string_validator(
                arg_name,
                min_length=validation_info.get("min_length"),
                max_length=validation_info.get("max_length"),
                pattern=validation_info.get("pattern"),
            )
        elif arg_type == "boolean":
            return create_boolean_validator(arg_name)
        elif arg_type == "choice":
            choices = arg_info.get("choices", [])
            return create_choice_validator(arg_name, choices)
        else:
            logger.warning(f"Unknown argument type for {arg_name}: {arg_type}")
            return None

    except Exception as e:
        logger.error(f"Failed to create validator for {arg_name}: {e}")
        return None
