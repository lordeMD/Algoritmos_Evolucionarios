import os
from cvrp_model import CVRPInstance
from genetic_algorithm import GeneticAlgorithmCVRP
from visualization import plot_routes, plot_convergence

def main():
    print("==========================================================")
    print("   Otimização de Roteamento de Veículos Capacitado (CVRP) ")
    print("            usando Algoritmos Evolutivos em Python        ")
    print("==========================================================\n")
    
    # 1. Configurações da Instância
    NUM_CUSTOMERS = 30    # Número de clientes
    VEHICLE_CAPACITY = 50 # Capacidade máxima de carga de cada veículo
    MAX_DEMAND = 10       # Demanda máxima de um cliente (em kg/unidades)
    INSTANCE_SEED = 42    # Semente para gerar os clientes de forma determinística
    
    print(f"1. Gerando instância sintética do problema...")
    print(f"   - Clientes: {NUM_CUSTOMERS}")
    print(f"   - Capacidade do Veículo: {VEHICLE_CAPACITY}")
    print(f"   - Demanda Máxima por Cliente: {MAX_DEMAND}")
    
    instance = CVRPInstance.generate_random_instance(
        num_customers=NUM_CUSTOMERS,
        capacity=VEHICLE_CAPACITY,
        max_demand=MAX_DEMAND,
        seed=INSTANCE_SEED
    )
    print("   Matriz de distâncias e demandas calculada com sucesso.\n")
    
    # 2. Configurações do Algoritmo Genético
    POP_SIZE = 150                      # Tamanho da população (indivíduos)
    CROSSOVER_RATE = 0.85               # Chance de cruzamento ordenado (OX)
    MUTATION_RATE = 0.25                # Chance de mutação (Inversão ou Swap)
    ELITISM_SIZE = 4                    # Quantidade de melhores indivíduos preservados
    MAX_GENERATIONS = 300               # Limite máximo de gerações
    EARLY_STOPPING_GENS = 60            # Critério de parada por estagnação
    GA_SEED = 123                       # Semente para os operadores aleatórios do AG
    
    print(f"2. Configurando Algoritmo Genético...")
    print(f"   - População: {POP_SIZE}")
    print(f"   - Taxa de Crossover: {CROSSOVER_RATE * 100}%")
    print(f"   - Taxa de Mutação: {MUTATION_RATE * 100}%")
    print(f"   - Tamanho do Elitismo: {ELITISM_SIZE}")
    print(f"   - Máximo de Gerações: {MAX_GENERATIONS}")
    print(f"   - Parada por Estagnação: {EARLY_STOPPING_GENS} gerações\n")
    
    ga = GeneticAlgorithmCVRP(
        instance=instance,
        pop_size=POP_SIZE,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        elitism_size=ELITISM_SIZE,
        seed=GA_SEED
    )
    
    # 3. Execução do Algoritmo
    print("3. Executando algoritmo evolucionário...")
    results = ga.evolve(generations=MAX_GENERATIONS, early_stopping_generations=EARLY_STOPPING_GENS)
    
    # 4. Exibição e Salvamento dos Resultados
    best_routes = results["best_routes"]
    best_cost = results["best_cost"]
    
    print("\n=================== RESULTADOS FINAIS ===================")
    print(f"Melhor Custo (Distância Total): {best_cost:.2f} km")
    print(f"Número de Veículos Utilizados: {len(best_routes)}")
    print("---------------------------------------------------------")
    
    for idx, route in enumerate(best_routes):
        # Calcula a demanda total daquela rota
        route_demand = sum(instance.demands[c] for c in route)
        # Calcula a distância daquela rota específica
        route_dist = ga.decoder._calculate_route_distance(route)
        print(f"Rota do Veículo {idx + 1:2d}: Depósito -> " + 
              " -> ".join(f"C{c}" for c in route) + 
              f" -> Depósito | Carga: {route_demand:2.0f}/{VEHICLE_CAPACITY} | Distância: {route_dist:6.2f} km")
    print("=========================================================\n")
    
    # 5. Geração de Gráficos e Salvamento em Arquivos
    print("5. Gerando gráficos de resultados...")
    
    # Criar caminhos para salvar
    routes_img_path = "cvrp_rotas_otimizadas.png"
    conv_img_path = "cvrp_convergencia.png"
    
    # Plota e salva as rotas
    plot_routes(instance, best_routes, save_path=routes_img_path)
    # Plota e salva a convergência
    plot_convergence(results["history_best"], results["history_avg"], save_path=conv_img_path)
    
    print(f"Gráficos salvos com sucesso na pasta do projeto!")

if __name__ == "__main__":
    main()
