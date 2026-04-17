param(
    [string]$m = "update",
    [switch]$NoPush
)

$repo = "Aircraft_Specific_Revenue_Roadmap"
$vpsPath = "/root/.openclaw/workspace-tag_coding/$repo"
$vpsHost = "root@185.164.110.65"
$sshKey = "/home/honeybadger/.ssh/id_ed25519"

# --- Git add, commit, push ---
git add -A
git commit -m $m
if (-not $NoPush) {
    git push origin main
}

# --- SSH to VPS: pull and rebuild ---
$sshCmd = @"
cd $vpsPath && git pull origin main && docker compose build --no-cache && docker compose up -d
"@

wsl ssh -i $sshKey $vpsHost $sshCmd
