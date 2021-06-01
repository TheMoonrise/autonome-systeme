@echo off
echo Running Flake8 (Linting)
py -m flake8 main
echo Running PyTest (Unit Testing)
py -m pytest main