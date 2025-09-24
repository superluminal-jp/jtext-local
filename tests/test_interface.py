"""
Tests for interface layer components.
"""

import pytest
from unittest.mock import Mock, patch

from jtext.interface import CLI, create_cli


class TestCLI:
    """Test CLI interface."""

    def test_cli_creation(self):
        """Test CLI creation."""
        cli = CLI()
        assert cli is not None
        assert cli.app is not None

    def test_create_cli_function(self):
        """Test create_cli function."""
        cli_app = create_cli()
        assert cli_app is not None

    @patch("click.echo")
    def test_cli_health_command(self, mock_echo):
        """Test CLI health command."""
        from click.testing import CliRunner

        runner = CliRunner()
        cli_app = create_cli()

        result = runner.invoke(cli_app, ["health"])
        assert result.exit_code == 0

        # Verify that health check output was called
        mock_echo.assert_called()

    @patch("click.echo")
    def test_cli_stats_command(self, mock_echo):
        """Test CLI stats command."""
        from click.testing import CliRunner

        runner = CliRunner()
        cli_app = create_cli()

        result = runner.invoke(cli_app, ["stats"])
        assert result.exit_code == 0

        # Verify that stats output was called
        mock_echo.assert_called()
