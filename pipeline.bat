@echo off
echo Running Flake8 (Linting)
python -m flake8 "%~dp0\main"
echo Running PyTest (Unit Testing)
python -m pytest "%~dp0\main"