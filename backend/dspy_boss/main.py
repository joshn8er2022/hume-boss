
"""
Main entry point for DSPY Boss system
"""

import asyncio
import argparse
import sys
from pathlib import Path
from loguru import logger

from .boss import run_dspy_boss


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DSPY Boss - Autonomous Agent Management System")
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Configuration directory path (default: configs)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode for testing"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup basic logging for startup
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=args.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    # Validate config directory
    config_path = Path(args.config_dir)
    if not config_path.exists():
        logger.info(f"Config directory {config_path} does not exist, it will be created with sample configs")
    
    try:
        # Run the system
        asyncio.run(run_dspy_boss(args.config_dir, args.dry_run))
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
