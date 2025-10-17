import os
import logging
from typing import Dict, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("The 'google-genai' library is required. Please install it using 'pip install google-genai'.")

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase

logger = logging.getLogger(__name__)


class GeminiLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "gemini-2.0-flash"

        api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        
        # 初始化token统计
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": None,
                "tool_calls": [],
            }

            # Extract content from the first candidate
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        processed_response["content"] = part.text
                        break

            # Extract function calls
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fn = part.function_call
                        processed_response["tool_calls"].append(
                            {
                                "name": fn.name,
                                "arguments": dict(fn.args) if fn.args else {},
                            }
                        )

            return processed_response
        else:
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        return part.text
            return ""

    def _reformat_messages(self, messages: List[Dict[str, str]]):
        """
        Reformat messages for Gemini.

        Args:
            messages: The list of messages provided in the request.

        Returns:
            tuple: (system_instruction, contents_list)
        """
        system_instruction = None
        contents = []

        for message in messages:
            if message["role"] == "system":
                system_instruction = message["content"]
            else:
                content = types.Content(
                    parts=[types.Part(text=message["content"])],
                    role=message["role"],
                )
                contents.append(content)

        return system_instruction, contents

    def _reformat_tools(self, tools: Optional[List[Dict]]):
        """
        Reformat tools for Gemini.

        Args:
            tools: The list of tools provided in the request.

        Returns:
            list: The list of tools in the required format.
        """

        def remove_additional_properties(data):
            """Recursively removes 'additionalProperties' from nested dictionaries."""
            if isinstance(data, dict):
                filtered_dict = {
                    key: remove_additional_properties(value)
                    for key, value in data.items()
                    if not (key == "additionalProperties")
                }
                return filtered_dict
            else:
                return data

        if tools:
            function_declarations = []
            for tool in tools:
                func = tool["function"].copy()
                cleaned_func = remove_additional_properties(func)

                function_declaration = types.FunctionDeclaration(
                    name=cleaned_func["name"],
                    description=cleaned_func.get("description", ""),
                    parameters=cleaned_func.get("parameters", {}),
                )
                function_declarations.append(function_declaration)

            tool_obj = types.Tool(function_declarations=function_declarations)
            return [tool_obj]
        else:
            return None

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        Generate a response based on the given messages using Gemini.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format for the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        
        # 获取调用者信息
        import inspect
        caller_frame = inspect.currentframe().f_back
        caller_name = caller_frame.f_code.co_name if caller_frame else "unknown"
        caller_module = caller_frame.f_globals.get('__name__', 'unknown') if caller_frame else "unknown"

        # Extract system instruction and reformat messages
        system_instruction, contents = self._reformat_messages(messages)

        # Prepare generation config
        config_params = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        # Add system instruction to config if present
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        if response_format is not None and response_format["type"] == "json_object":
            config_params["response_mime_type"] = "application/json"
            if "schema" in response_format:
                config_params["response_schema"] = response_format["schema"]

        if tools:
            formatted_tools = self._reformat_tools(tools)
            config_params["tools"] = formatted_tools

            if tool_choice:
                if tool_choice == "auto":
                    mode = types.FunctionCallingConfigMode.AUTO
                elif tool_choice == "any":
                    mode = types.FunctionCallingConfigMode.ANY
                else:
                    mode = types.FunctionCallingConfigMode.NONE

                tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=mode,
                        allowed_function_names=(
                            [tool["function"]["name"] for tool in tools] if tool_choice == "any" else None
                        ),
                    )
                )
                config_params["tool_config"] = tool_config

        generation_config = types.GenerateContentConfig(**config_params)

        response = self.client.models.generate_content(
            model=self.config.model, contents=contents, config=generation_config
        )
        
        ret = self._parse_response(response, tools)
        
        # 提取并统计token使用信息（包含调用者信息和响应内容）
        self._log_token_usage(response, caller_name, caller_module, ret, tools)
        
        if ret is None or ret == "":
            # 打印状态码和错误代码
            for i in range(len(response.candidates)):
                print(f"Candidate {i}:")
                print(f"  Status code: {response.candidates[i].finish_reason}")
            raise ValueError("Empty response from Gemini API")

        return ret
    
    def _log_token_usage(self, response, caller_name: str, caller_module: str, output_content, tools=None):
        """
        提取并记录token使用统计信息
        
        Args:
            response: Gemini API响应对象
            caller_name: 调用者函数名
            caller_module: 调用者模块名
            output_content: LLM的响应输出内容
            tools: 使用的工具列表
        """
        try:
            # 增加调用计数
            self.call_count += 1
            
            # 从response中提取token使用信息
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                
                # 提取token数量
                input_tokens = getattr(usage, 'prompt_token_count', 0)
                output_tokens = getattr(usage, 'candidates_token_count', 0)
                total = getattr(usage, 'total_token_count', 0)
                
                # 累加到总计
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_tokens += total
                
                # 确定调用类型
                call_type = "Tool Calling" if tools else "Text Generation"
                
                # 格式化输出内容（截断过长的内容）
                if isinstance(output_content, dict):
                    # Tool calling响应
                    if 'tool_calls' in output_content and output_content['tool_calls']:
                        tool_names = [tc['name'] for tc in output_content['tool_calls']]
                        output_preview = f"Tool调用: {', '.join(tool_names)}"
                        if output_content.get('content'):
                            output_preview += f" | 内容: {str(output_content['content'])[:100]}"
                    else:
                        output_preview = str(output_content)[:150]
                elif isinstance(output_content, str):
                    output_preview = output_content[:150]
                    if len(output_content) > 150:
                        output_preview += "..."
                else:
                    output_preview = str(output_content)[:150]
                
                # 打印详细信息到console
                print("\n" + "="*80)
                print(f"🔷 Gemini API调用 #{self.call_count}")
                print("="*80)
                print(f"📍 调用来源: {caller_module}.{caller_name}()")
                print(f"🏷️  调用类型: {call_type}")
                print(f"📥 输入tokens: {input_tokens:,}")
                print(f"📤 输出tokens: {output_tokens:,}")
                print(f"📊 本次合计: {total:,} tokens")
                print(f"📈 累计总tokens: {self.total_tokens:,}")
                print(f"💬 输出内容预览:")
                print(f"   {output_preview}")
                print("="*80 + "\n")
                
                # 记录到logger（更详细）
                logger.info(f"🔷 Gemini API调用 #{self.call_count}")
                logger.info(f"  📍 调用来源: {caller_module}.{caller_name}()")
                logger.info(f"  🏷️  调用类型: {call_type}")
                logger.info(f"  📥 输入tokens: {input_tokens:,}")
                logger.info(f"  📤 输出tokens: {output_tokens:,}")
                logger.info(f"  📊 本次总计: {total:,}")
                logger.info(f"  📈 累计总输入: {self.total_input_tokens:,}")
                logger.info(f"  📈 累计总输出: {self.total_output_tokens:,}")
                logger.info(f"  📈 累计总tokens: {self.total_tokens:,}")
                logger.info(f"  💬 输出内容: {output_content}")
            else:
                print(f"⚠️  调用#{self.call_count} ({caller_module}.{caller_name}): 响应中未找到usage_metadata")
                logger.warning(f"⚠️  Gemini API响应中未找到usage_metadata，无法统计token")
                
        except Exception as e:
            logger.error(f"❌ 统计token使用时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def get_token_stats(self) -> Dict[str, int]:
        """
        获取token使用统计
        
        Returns:
            dict: 包含token统计信息的字典
        """
        return {
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "average_input_tokens": self.total_input_tokens // self.call_count if self.call_count > 0 else 0,
            "average_output_tokens": self.total_output_tokens // self.call_count if self.call_count > 0 else 0,
        }
    
    def print_token_summary(self):
        """打印token使用总结"""
        stats = self.get_token_stats()
        print("\n" + "="*60)
        print("📊 Gemini Token使用总结")
        print("="*60)
        print(f"🔢 总调用次数: {stats['call_count']:,}")
        print(f"📥 总输入tokens: {stats['total_input_tokens']:,}")
        print(f"📤 总输出tokens: {stats['total_output_tokens']:,}")
        print(f"📊 总计tokens: {stats['total_tokens']:,}")
        if stats['call_count'] > 0:
            print(f"📊 平均输入tokens: {stats['average_input_tokens']:,}")
            print(f"📊 平均输出tokens: {stats['average_output_tokens']:,}")
        print("="*60 + "\n")
