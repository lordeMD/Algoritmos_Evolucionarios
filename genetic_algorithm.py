import numpy as np
from cvrp_model import RouteDecoder

class GeneticAlgorithmCVRP:

    def __init__(self, instance, pop_size=100, crossover_rate=0.8, 
                 mutation_rate=0.2, elitism_size=2, seed=42):

        self.instance = instance
        self.decoder = RouteDecoder(instance)
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_size = elitism_size
        
        # Semente do gerador aleatório do NumPy para garantir reprodutibilidade
        np.random.seed(seed)
        
    def initialize_population(self):

        population = []
        customers = list(range(1, self.instance.num_customers + 1))
        
        for _ in range(self.pop_size):
            # Cria uma permutação aleatória dos clientes
            ind = list(np.random.permutation(customers))
            population.append(ind)
            
        return population

    def evaluate_population(self, population):

        costs = np.zeros(self.pop_size)
        decoded_routes = []
        
        for idx, ind in enumerate(population):
            routes, dist = self.decoder.decode(ind)
            costs[idx] = dist
            decoded_routes.append(routes)
            
        return costs, decoded_routes

    def selection_tournament(self, population, costs, k=3):

        # Escolhe K índices aleatórios da população
        selected_indices = np.random.choice(self.pop_size, size=k, replace=False)
        
        # Identifica o índice do indivíduo com o menor custo (melhor aptidão)
        best_idx = selected_indices[np.argmin(costs[selected_indices])]
        
        # Retorna uma cópia do cromossomo selecionado
        return list(population[best_idx])

    def crossover_ox(self, parent1, parent2):

        n = len(parent1)
        
        def generate_child(p1, p2):
            child = [-1] * n
            
            # 1. Seleciona dois pontos de corte aleatórios
            idx1 = np.random.randint(0, n - 1)
            idx2 = np.random.randint(idx1 + 1, n)
            
            # 2. Copia o segmento do p1 para o filho
            child[idx1:idx2] = p1[idx1:idx2]
            
            # 3. Preenche o restante do filho usando a ordem relativa do p2
            # Começamos a preencher logo após o segundo corte (ponto idx2)
            current_pos = idx2
            
            # Cicla sobre os elementos do p2 começando a partir de idx2
            p2_ordered = p2[idx2:] + p2[:idx2]
            
            for item in p2_ordered:
                if item not in child:
                    # Se alcançamos o final do vetor, voltamos ao início (circular)
                    if current_pos >= n:
                        current_pos = 0
                    child[current_pos] = item
                    current_pos += 1
            return child

        child1 = generate_child(parent1, parent2)
        child2 = generate_child(parent2, parent1)
        
        return child1, child2

    def mutate_inversion(self, individual):

        n = len(individual)
        idx1 = np.random.randint(0, n - 1)
        idx2 = np.random.randint(idx1 + 1, n)
        
        # Inverte o segmento selecionado
        individual[idx1:idx2] = individual[idx1:idx2][::-1]

    def mutate_swap(self, individual):

        n = len(individual)
        i, j = np.random.choice(n, size=2, replace=False)
        individual[i], individual[j] = individual[j], individual[i]

    def evolve(self, generations=100, early_stopping_generations=50):

        # 1. Inicializa população
        population = self.initialize_population()
        
        # Históricos de métricas para plotagem e análise
        history_best = []
        history_avg = []
        
        # Variáveis de controle do melhor indivíduo global
        global_best_cost = float('inf')
        global_best_chromosome = None
        global_best_routes = None
        generations_without_improvement = 0
        
        print("Iniciando a evolução...")
        
        for gen in range(1, generations + 1):
            # 2. Avaliação da população atual
            costs, decoded_routes = self.evaluate_population(population)
            
            # Encontra o melhor indivíduo desta geração
            best_gen_idx = np.argmin(costs)
            best_gen_cost = costs[best_gen_idx]
            avg_gen_cost = np.mean(costs)
            
            history_best.append(best_gen_cost)
            history_avg.append(avg_gen_cost)
            
            # Atualiza o melhor global, se aplicável
            if best_gen_cost < global_best_cost:
                global_best_cost = best_gen_cost
                global_best_chromosome = list(population[best_gen_idx])
                global_best_routes = decoded_routes[best_gen_idx]
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
                
            # Log de progresso a cada 10 gerações ou na primeira/última
            if gen == 1 or gen % 10 == 0 or gen == generations:
                print(f"Geração {gen:3d} | Melhor Custo: {best_gen_cost:7.2f} | Custo Médio: {avg_gen_cost:7.2f} | Sem Melhorias: {generations_without_improvement:3d}")
            
            # Critério de parada por estagnação
            if generations_without_improvement >= early_stopping_generations:
                print(f"\nCritério de parada atingido na geração {gen}: {early_stopping_generations} gerações consecutivas sem melhoria no melhor custo.")
                break
                
            # 3. Construção da nova geração
            new_population = []
            
            # 3.1 Elitismo: Preserva os E melhores indivíduos diretamente
            sorted_indices = np.argsort(costs)
            for idx in sorted_indices[:self.elitism_size]:
                new_population.append(list(population[idx]))
                
            # 3.2 Laço reprodutivo (cruzamento e mutação)
            while len(new_population) < self.pop_size:
                # Seleção dos pais
                parent1 = self.selection_tournament(population, costs)
                parent2 = self.selection_tournament(population, costs)
                
                # Crossover
                if np.random.rand() < self.crossover_rate:
                    child1, child2 = self.crossover_ox(parent1, parent2)
                else:
                    child1, child2 = list(parent1), list(parent2)
                    
                # Mutação (aplicada a cada filho de forma independente)
                for child in [child1, child2]:
                    if np.random.rand() < self.mutation_rate:
                        # 75% de probabilidade de mutação por inversão, 25% por troca (swap)
                        if np.random.rand() < 0.75:
                            self.mutate_inversion(child)
                        else:
                            self.mutate_swap(child)
                            
                    # Garante que adicionamos apenas até atingir pop_size
                    if len(new_population) < self.pop_size:
                        new_population.append(child)
                        
            population = new_population
            
        print("\nEvolução finalizada!")
        print(f"Melhor custo obtido: {global_best_cost:.2f}")
        
        return {
            "best_chromosome": global_best_chromosome,
            "best_routes": global_best_routes,
            "best_cost": global_best_cost,
            "history_best": history_best,
            "history_avg": history_avg
        }
