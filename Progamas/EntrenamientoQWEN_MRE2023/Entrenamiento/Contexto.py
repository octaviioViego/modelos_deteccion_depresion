from transformers import AutoTokenizer
                          

def enviar_contexto(mensaje, etiqueta):
    
    MODEL_ID = "Qwen/Qwen3.5-9B"
    MAX_MESSAGE_TOKENS = 256
    SYSTEM_PROMPT = ""
    MAX_SEQ_LENGTH = 384
    END_IDS = tokenizer.encode("<|im_end|>", 
                           add_special_tokens=False)


    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Limitamos el mensaje para controlar el consumo de memoria.
    message_ids = tokenizer.encode(mensaje,
                                   add_special_tokens=False,
                                   truncation=True,
                                   max_length=MAX_MESSAGE_TOKENS,)

    mensaje_recortado = tokenizer.decode(message_ids,
                                         skip_special_tokens=True,)

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user","content": f"Mensaje a clasificar:\n{mensaje_recortado}",},]

    prompt_ids = tokenizer.apply_chat_template(messages,
                                               tokenize=True,
                                               add_generation_prompt=True,
                                               enable_thinking=False,)

    # El objetivo es la etiqueta, seguida del fin del turno.
    target_ids = tokenizer.encode(str(int(etiqueta)),
                                  add_special_tokens=False,) + END_IDS

    input_ids = prompt_ids + target_ids

    if len(input_ids) > MAX_SEQ_LENGTH:
        raise ValueError(
            f"Secuencia de {len(input_ids)} tokens; "
            f"supera MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}. "
            "Reduce MAX_MESSAGE_TOKENS."
        )

    # No calculamos pérdida sobre el prompt, solo sobre la respuesta.
    labels = [-100] * len(prompt_ids) + target_ids

    return {"input_ids": input_ids,"labels": labels,}
