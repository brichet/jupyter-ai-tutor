import json
import tempfile
from datetime import datetime
from pathlib import Path

import tornado
from jupyter_server.base.handlers import APIHandler
from tornado.iostream import StreamClosedError


class ExplainHandler(APIHandler):
    @tornado.web.authenticated
    async def post(self):
        body = self.get_json_body()
        if not body or "body" not in body:
            raise tornado.web.HTTPError(400, "Missing 'body' field in request")

        message_body = body["body"]
        description = body.get("description", "")
        if description:
            message_body = (
                f"<exercise_description>\n{description}\n</exercise_description>\n\n"
                f"{message_body}"
            )

        config_manager = self.settings.get("jupyternaut.config_manager")
        if not config_manager:
            raise tornado.web.HTTPError(
                503, "Jupyternaut config manager is not available"
            )
        if not config_manager.chat_model:
            raise tornado.web.HTTPError(
                503,
                "No chat model is configured. Set one in 'Settings > AI Settings'.",
            )

        self.set_header("Content-Type", "text/event-stream")
        self.set_header("Cache-Control", "no-cache")
        self.set_header("X-Accel-Buffering", "no")

        system_prompt = self.settings.get("jupyter_ai_tutor.system_prompt", "")
        debug_mode = self.settings.get("jupyter_ai_tutor.debug", False)
        prompt_file = None
        answer_file = None
        accumulated_response = ""

        if debug_mode:
            debug_dir = Path(tempfile.gettempdir()) / "jupyter-ai-tutor"
            try:
                debug_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.log.error(f"Failed to create debug directory {debug_dir}: {e}")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            prompt_file = debug_dir / f"{timestamp}_jupyter_tutor_prompt.txt"
            answer_file = debug_dir / f"{timestamp}_jupyter_tutor_answer.txt"
            try:
                with prompt_file.open("w", encoding="utf-8") as f:
                    f.write("=== SYSTEM PROMPT ===\n\n")
                    f.write(system_prompt)
                    f.write("\n\n=== USER MESSAGE ===\n\n")
                    f.write(message_body)
            except Exception as e:
                self.log.error(f"Failed to write tutor debug prompt: {e}")

        try:
            from jupyter_ai_jupyternaut.jupyternaut.chat_models import ChatLiteLLM
            from langchain_core.messages import HumanMessage, SystemMessage

            model = ChatLiteLLM(
                **config_manager.chat_model_args,
                model=config_manager.chat_model,
                streaming=True,
            )

            async for chunk in model.astream(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=message_body),
                ]
            ):
                text = (
                    chunk.content
                    if isinstance(chunk.content, str)
                    else "".join(
                        block.get("text", "")
                        for block in chunk.content
                        if isinstance(block, dict)
                    )
                )
                if text:
                    if debug_mode:
                        accumulated_response += text
                    self.write(f"data: {json.dumps({'text': text})}\n\n")
                    self.flush()

            if debug_mode and answer_file:
                try:
                    with answer_file.open("w", encoding="utf-8") as f:
                        f.write(accumulated_response)
                        f.write("\n")
                except Exception as e:
                    self.log.error(f"Failed to write tutor debug answer: {e}")

        except StreamClosedError:
            if debug_mode and answer_file and accumulated_response:
                try:
                    with answer_file.open("w", encoding="utf-8") as f:
                        f.write(accumulated_response)
                        f.write("\n")
                except Exception as e:
                    pass
            return  # Client disconnected; stop streaming
        except Exception as e:
            self.log.exception("Error during tutor LLM call")
            try:
                self.write(f"data: {json.dumps({'error': str(e)})}\n\n")
                self.flush()
            except StreamClosedError:
                return

        try:
            self.write("data: [DONE]\n\n")
            self.flush()
            self.finish()
        except StreamClosedError:
            pass
