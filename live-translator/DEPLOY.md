# Развёртывание на своём сервере

Инструкция для человека или ИИ-ассистента (например, Claude Code на сервере).

**Что получится:** веб-приложение по адресу `https://<домен>`, которое открывается с телефона. Внутри два контейнера: `app` (Node.js, порт 8080) и `caddy` (HTTPS-сертификат Let's Encrypt, выдаётся автоматически, плюс проксирование WebSocket).

HTTPS обязателен: без него телефон не даст доступ к микрофону.

## Требования
- Linux-сервер с публичным IP, Docker и Docker Compose v2 (`docker compose version`).
- Открыты входящие порты **80** и **443**.
- Домен, указывающий на IP сервера. Если домена нет, подойдёт `<IP>.sslip.io`, например `203.0.113.10.sslip.io`: этот адрес сам указывает на IP, сертификат Caddy получит автоматически.

## Установка

```bash
git clone --branch ccr-f259f1e1-3rh5e1 --depth 1 https://github.com/dave-in/testnets-1.git
cd testnets-1/live-translator
cp .env.example .env
chmod 600 .env
```

Заполните `.env`. **Ключи владелец вписывает сам**, например через `nano .env`. Не передавайте их в чат или ассистенту.

| Переменная | Значение |
|---|---|
| `DOMAIN` | домен или `<IP>.sslip.io` |
| `APP_PASSWORD` | пароль для входа в приложение (длинный) |
| `DASHSCOPE_API_KEY` | ключ Alibaba Model Studio (`sk-…` или `sk-ws-…`) |
| `QWEN_API_HOST` | для ключей `sk-ws-…`: «API Host» из окна создания ключа (`ws-….ap-southeast-1.maas.aliyuncs.com`) |
| `ELEVENLABS_API_KEY` | ключ ElevenLabs |
| `ELEVENLABS_VOICE_ID` | ID клонированного голоса |

Запуск:

```bash
docker compose up -d --build
docker compose logs -f app      # ждать строку "live-translator: http://localhost:8080"
docker compose logs caddy | grep -i certificate   # сертификат получен
curl -s https://$DOMAIN/healthz  # должно вернуть ok
curl -s https://$DOMAIN/api/config  # "qwen":true, "elevenlabs":true — ключи подхватились
```

## Если на сервере уже есть nginx или Caddy на портах 80/443

Не поднимайте контейнер `caddy`. Запустите только приложение и проксируйте на него из существующего веб-сервера:

```bash
docker compose up -d --build app
# пробросить порт наружу только на localhost: добавьте в сервис app
#   ports: ["127.0.0.1:8080:8080"]
```

Для nginx (важно передать заголовки WebSocket):

```nginx
location / {
    proxy_pass http://127.0.0.1:8080;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 3600s;
}
```

## Без Docker

Нужен Node.js 20+:

```bash
cd testnets-1/live-translator
npm ci --omit=dev
node --env-file=.env server.js      # слушает PORT (по умолчанию 8080)
```

Для автозапуска используйте systemd-юнит с `EnvironmentFile=/путь/к/.env` и `ExecStart=/usr/bin/node server.js`. HTTPS делается через свой nginx или Caddy, как в разделе выше.

## Обновление

```bash
cd testnets-1/live-translator
git pull
docker compose up -d --build
```

## Проверка

1. Откройте `https://<домен>` на телефоне и введите пароль в «Настройках».
2. Режим **Qwen**, язык **English**: нажмите на микрофон и скажите фразу по-русски. Должны появиться текст и голос перевода.
3. Режим **ElevenLabs**: то же самое, перевод должен звучать вашим клоном.
4. Если приложение показывает ошибку, ищите подробности в `docker compose logs app`:
   - `HTTP 401` или `403` от Qwen: неверный ключ, не задан `QWEN_API_HOST` для ключа `sk-ws-…`, у ключа нет доступа к нужной модели или не пополнен баланс.
   - Ошибка модели Qwen: попробуйте `QWEN_MODEL=qwen3-livetranslate-flash-realtime`.
   - Ошибка ElevenLabs про модель или права: `ELEVENLABS_MODEL=eleven_flash_v2_5`, либо дайте ключу доступ к Text to Dialogue.
   - После правки `.env` выполните `docker compose up -d`.
