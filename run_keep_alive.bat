@echo off
title LunarMatch-TGS Keep-Alive Service
echo Starting Keep-Alive ping service for Render backend...
python keep_alive.py https://lunarmatch-tgs.onrender.com 600
pause
