from ultralytics import YOLO

# Carregue o seu modelo. 
try:
    # Altere conforme o modelo que estiver utilizando 
    model = YOLO('yolov10n.pt')

    # Analise das classes para as quais o modelo foi treinado
    print("OK! Informações das Classes:")
    if model.names:
        print(f"   - O modelo foi treinado para detectar {len(model.names)} classes.")
        # Imprime o mapeamento de ID para nome da classe
        print("   - Mapeamento (ID -> Nome):", model.names) 
    else:
        print("   - Não foi possível encontrar os nomes das classes no modelo.")

    print("\n" + "="*50 + "\n")

    # Analise da arquitetura e os hiperparâmetros (se disponíveis)
    print("OK! Informações da Arquitetura e Treinamento:")
    # A própria representação do objeto do modelo já mostra a arquitetura
    print(model) 
    
    # Argumentos de treinamento podem estar salvos no modelo
    if hasattr(model, 'args'):
        print("\n   - Argumentos de treinamento salvos:")
        # Imprime os argumentos de forma legível
        for key, value in model.args.items():
            print(f"     {key}: {value}")

except Exception as e:
    print(f"BAD! Ocorreu um erro ao carregar o modelo: {e}")
    print("   - Verifique se o arquivo está na pasta correta.")
    print("   - Este método funciona melhor com modelos treinados com a biblioteca 'ultralytics'.")