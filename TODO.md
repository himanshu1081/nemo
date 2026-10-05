# TODO

## Blocking

- [ ] Add `ALEXA_SKILL_ID` to `.env` (Alexa Developer Console, `amzn1.ask.skill...`). Without it every Alexa request is rejected with 400.
- [ ] Add `NEMO_USER_ID` to `.env`, the user ID Gmail was connected under. Without it Gmail actions fail.
- [ ] Connect Gmail once through the `/api/connectors` OAuth flow (`connector_info` is empty), using the same user ID as `NEMO_USER_ID`.

## Alexa Developer Console

- [ ] Set the endpoint to `https://<host>/alexa`.
- [ ] Interaction model: `ChatIntent` with a `message` slot (`AMAZON.SearchQuery`), plus Yes, No, Stop, Cancel and Help intents.
- [ ] Subscribe to the `SkillDisabled` event in the skill manifest so memory is wiped when a user disables the skill.

## Deploy

- [ ] Install the updated `requirements.txt` (includes `fastembed`) on the host.
- [ ] Make sure `ALEXA_SKIP_VERIFICATION` is not set in production.
- [ ] Expect slower cold starts while the embedding model downloads on first boot.

## Later

- [ ] Replace the single-user `NEMO_USER_ID` with Alexa Account Linking (`resolve_user_id` in `routes/chat.py`).
- [ ] Add the web app's production domain to the CORS `allow_origins` in `main.py`.
