#!/usr/bin/env python3
"""
Script to test database connection
"""
import sys
import os
import logging
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.config import settings
from app.database.connection import test_connection, engine
from app.database.session import db_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_database_configuration() -> Dict[str, Any]:
    """
    Test database configuration and display connection details
    """
    logger.info("Testing database configuration...")
    
    config_info = {
        "host": settings.db_host,
        "port": settings.db_port,
        "database": settings.db_name,
        "user": settings.db_user,
        "url_masked": f"postgresql://{settings.db_user}:****@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    }
    
    logger.info(f"Database configuration:")
    for key, value in config_info.items():
        logger.info(f"  {key}: {value}")
    
    return config_info


def test_basic_connection() -> bool:
    """
    Test basic database connection
    """
    logger.info("Testing basic database connection...")
    
    try:
        success = test_connection()
        if success:
            logger.info("✅ Basic database connection successful")
        else:
            logger.error("❌ Basic database connection failed")
        return success
    except Exception as e:
        logger.error(f"❌ Exception during basic connection test: {e}")
        return False


def test_session_manager() -> bool:
    """
    Test database session manager
    """
    logger.info("Testing database session manager...")
    
    try:
        health_status = db_manager.health_check()
        if health_status:
            logger.info("✅ Database session manager working correctly")
        else:
            logger.error("❌ Database session manager health check failed")
        return health_status
    except Exception as e:
        logger.error(f"❌ Exception during session manager test: {e}")
        return False


def test_engine_info():
    """
    Display SQLAlchemy engine information
    """
    logger.info("SQLAlchemy Engine Information:")
    logger.info(f"  Engine: {engine}")
    logger.info(f"  URL: {engine.url}")
    logger.info(f"  Driver: {engine.url.drivername}")
    logger.info(f"  Pool size: {engine.pool.size()}")
    logger.info(f"  Pool checked out: {engine.pool.checkedout()}")


def run_all_tests() -> bool:
    """
    Run all database tests
    """
    logger.info("=" * 60)
    logger.info("Database Connection Test Suite")
    logger.info("=" * 60)
    
    all_passed = True
    
    # Test 1: Configuration
    try:
        test_database_configuration()
        logger.info("✅ Configuration test passed")
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        all_passed = False
    
    print()
    
    # Test 2: Engine info
    try:
        test_engine_info()
        logger.info("✅ Engine info test passed")
    except Exception as e:
        logger.error(f"❌ Engine info test failed: {e}")
        all_passed = False
    
    print()
    
    # Test 3: Basic connection
    try:
        if not test_basic_connection():
            all_passed = False
    except Exception as e:
        logger.error(f"❌ Basic connection test failed: {e}")
        all_passed = False
    
    print()
    
    # Test 4: Session manager
    try:
        if not test_session_manager():
            all_passed = False
    except Exception as e:
        logger.error(f"❌ Session manager test failed: {e}")
        all_passed = False
    
    print()
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("🎉 All database tests passed!")
        logger.info("Database integration is ready to use.")
    else:
        logger.error("❌ Some database tests failed!")
        logger.error("Please check your database configuration and ensure PostgreSQL is running.")
    
    logger.info("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test database connection")
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run only basic connection test"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        success = test_basic_connection()
        sys.exit(0 if success else 1)
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
