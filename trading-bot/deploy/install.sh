#!/usr/bin/env bash
# Установка сигнального бота на сервер одной командой:
#
#   curl -fsSL https://raw.githubusercontent.com/Dave-in/testnets-1/claude/trading-bot-exchanges-pp8kid/trading-bot/deploy/install.sh | bash
#
# Сайт и веб-сервер не затрагиваются: бот не открывает портов, живёт в своей папке
# (~/trading-bot-app), в своём окружении Python и своём сервисе systemd «trading-bot».
# Повторный запуск обновляет код и перезапускает бота; счёт и настройки сохраняются.
set -euo pipefail

REPO="${BOT_REPO:-https://github.com/Dave-in/testnets-1.git}"
BRANCH="${BOT_BRANCH:-claude/trading-bot-exchanges-pp8kid}"
DIR="${BOT_DIR:-$HOME/trading-bot-app}"
SERVICE=trading-bot
# По умолчанию запускаются сразу три виртуальных счёта из bots.toml (для сравнения стратегий):
# «Спот-быстрый» (основной), «Фьючерсы» (лонги и шорты) и «Спот-база» (без сигналов).
# Только один счёт: BOT_CONFIG=config.trend.toml (или другой config.*.toml) перед командой.
if [ -n "${BOT_CONFIG:-}" ]; then
  RUN="-m bot --config $BOT_CONFIG run"; WHAT="$BOT_CONFIG"
else
  RUN="-m bot run-all"; WHAT="три виртуальных счёта (bots.toml)"
fi
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
die() { printf '\nОШИБКА: %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null || die "нужен git:  sudo apt install -y git"
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then PY="$c"; break; fi
done
[ -n "$PY" ] || die "нужен Python 3.10 или новее (сейчас: $(python3 --version 2>&1 || echo нет))"

say "1/4 Код → $DIR"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch -q --depth 1 origin "$BRANCH" && git -C "$DIR" checkout -q -B "$BRANCH" FETCH_HEAD
else
  git clone -q --depth 1 -b "$BRANCH" "$REPO" "$DIR"
fi
cd "$DIR/trading-bot"

say "2/4 Окружение Python ($($PY --version)) и библиотеки"
[ -x .venv/bin/python ] || "$PY" -m venv .venv \
  || die "не создаётся окружение. На Ubuntu/Debian:  sudo apt install -y python3-venv"
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-signals.txt

say "3/4 Telegram"
touch .env && chmod 600 .env
if ! grep -q '^TELEGRAM_BOT_TOKEN=.\+' .env; then
  TOKEN="${TELEGRAM_BOT_TOKEN:-}"
  if [ -z "$TOKEN" ]; then
    read -rsp "Вставьте токен бота от @BotFather (ввод скрыт) и нажмите Enter: " TOKEN </dev/tty; echo
  fi
  [ -n "$TOKEN" ] || die "токен пустой"
  sed -i '/^TELEGRAM_BOT_TOKEN=/d' .env && printf 'TELEGRAM_BOT_TOKEN=%s\n' "$TOKEN" >> .env
fi
if ! grep -q '^TELEGRAM_CHAT_ID=.\+' .env; then
  if [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
    printf 'TELEGRAM_CHAT_ID=%s\n' "$TELEGRAM_CHAT_ID" >> .env
  else
    .venv/bin/python -m bot tg-setup --wait 600 \
      || die "не удалось связаться с вами в Telegram — отправьте боту /start и запустите установку ещё раз"
  fi
fi

say "4/4 Запуск"
if [ "${BOT_NO_SERVICE:-0}" = 1 ]; then
  echo "Сервис не ставлю (BOT_NO_SERVICE=1). Ручной запуск:"
  echo "  cd $DIR/trading-bot && .venv/bin/python $RUN"
  exit 0
fi
command -v systemctl >/dev/null || die "нет systemd. Запустите вручную в tmux:  cd $DIR/trading-bot && .venv/bin/python $RUN"
SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"
$SUDO tee /etc/systemd/system/$SERVICE.service >/dev/null <<UNIT
[Unit]
Description=Crypto signal bot (paper trading + Telegram)
After=network-online.target
Wants=network-online.target

[Service]
User=$(id -un)
WorkingDirectory=$DIR/trading-bot
ExecStart=$DIR/trading-bot/.venv/bin/python $RUN
Restart=always
RestartSec=15

[Install]
WantedBy=multi-user.target
UNIT
$SUDO systemctl daemon-reload
$SUDO systemctl enable -q $SERVICE
$SUDO systemctl restart $SERVICE
sleep 5
$SUDO systemctl is-active -q $SERVICE || { $SUDO journalctl -u $SERVICE -n 30 --no-pager; die "бот не запустился, лог выше"; }

say "Готово: запущено — $WHAT. Бот переживёт перезагрузку сервера."
cat <<TXT
В Telegram должно прийти «Бот запущен». Команды боту: /status /report /pause /resume /closeall
На сервере:
  sudo systemctl status $SERVICE      — состояние
  sudo journalctl -u $SERVICE -f      — живой лог
  sudo systemctl stop $SERVICE        — остановить
  повторить эту установку             — обновить бота
TXT
