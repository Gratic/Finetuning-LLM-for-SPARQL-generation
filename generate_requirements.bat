@echo off
SETLOCAL
set file=requirements.txt

echo WARNING: Dependencies inside of Jupyter Notebooks will not be fetch. Please convert notebooks to python file before generating %file%.

echo Generating %file%...
echo # Required for the auto generation of %file% (this file) through %~nx0 > %file%
echo pipreqs >> %file%
echo. >> %file%
echo # Dependencies from python file starts below >> %file%
echo --find-links https://download.pytorch.org/whl/torch_stable.html >> %file%
echo. >> %file%
python.exe -m pipreqs.pipreqs --encoding utf8 --ignore .venv,outputs,.git . --print >> %file%
echo Generation done.
ENDLOCAL
PAUSE