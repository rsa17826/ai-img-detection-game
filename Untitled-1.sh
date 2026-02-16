sudo fallocate -l 40G /ssd/swapfile2
sudo chmod 600 /ssd/swapfile2
sudo mkswap /ssd/swapfile2
sudo swapon /ssd/swapfile2

gsettings set org.gnome.desktop.wm.keybindings switch-to-workspace-up "['']"
gsettings set org.gnome.desktop.wm.keybindings switch-to-workspace-down "['']"

kwriteconfig6 --file kglobalshortcutsrc --group kwin --key "Switch One Desktop Up" "none,none,Switch One Desktop Up"
kwriteconfig6 --file kglobalshortcutsrc --group kwin --key "Switch One Desktop Down" "none,none,Switch One Desktop Down"

# Reload the shortcut daemon to apply changes
kquitapp6 kglobalaccel && sleep 2s && kglobalaccel6 &

sudo systemctl stop nvargus-daemon
sudo systemctl disable nvargus-daemon
sudo rm /tmp/argus_socket
sudo systemctl enable nvargus-daemon
sudo systemctl start nvargus-daemon