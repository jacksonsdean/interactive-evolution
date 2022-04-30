#!bin/sh
cd "`dirname $0`"
git pull
source ./venv/bin/activate
pip install -r requirements.txt
chmod +x update.command
chmod +x run.command