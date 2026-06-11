import numpy as np

class CVRPInstance:
    """
    Classe que modela uma instância do Problema de Roteamento de Veículos Capacitado (CVRP).
    Esta classe armazena as coordenadas dos clientes, suas respectivas demandas,
    a capacidade máxima do veículo e calcula a matriz de distâncias.
    """
    def __init__(self, coords, demands, capacity):
        """
        Inicializa a instância do CVRP.
        
        Args:
            coords (np.ndarray): Matriz de coordenadas (N+1, 2), onde a linha 0 é o depósito
                                 e as linhas 1 a N são as coordenadas dos clientes.
            demands (np.ndarray): Vetor de demandas (N+1,), onde a posição 0 é 0 (depósito)
                                  e as posições 1 a N são as demandas dos clientes.
            capacity (float): Capacidade máxima de carga de cada veículo.
        """
        self.coords = np.array(coords, dtype=float)
        self.demands = np.array(demands, dtype=float)
        self.capacity = float(capacity)
        self.num_customers = len(coords) - 1
        
        # Calcula a matriz de distâncias euclidianas entre todos os pontos
        self.dist_matrix = self._calculate_dist_matrix()
        
    def _calculate_dist_matrix(self):
        """Calcula a matriz de distâncias euclidianas."""
        n_points = len(self.coords)
        dist_matrix = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # Distância euclidiana: sqrt((x1-x2)^2 + (y1-y2)^2)
                    dist_matrix[i, j] = np.linalg.norm(self.coords[i] - self.coords[j])
        return dist_matrix

    @classmethod
    def generate_random_instance(cls, num_customers=30, capacity=50, max_demand=10, seed=42):
        """
        Método de fábrica (Factory Method) para gerar uma instância sintética e realista do CVRP.
        
        Args:
            num_customers (int): Número de clientes a serem atendidos.
            capacity (float): Capacidade máxima do veículo.
            max_demand (int): Demanda máxima por cliente.
            seed (int): Semente aleatória para fins de reprodutibilidade.
            
        Returns:
            CVRPInstance: Uma instância configurada do problema.
        """
        np.random.seed(seed)
        
        # 1. Coordenadas: Depósito no centro (50, 50), clientes distribuídos aleatoriamente em [0, 100]x[0, 100]
        coords = np.zeros((num_customers + 1, 2))
        coords[0] = [50.0, 50.0]  # Depósito
        coords[1:] = np.random.uniform(0.0, 100.0, size=(num_customers, 2))
        
        # 2. Demandas: Demanda do depósito é 0. Clientes têm demandas aleatórias entre 1 e max_demand
        demands = np.zeros(num_customers + 1)
        demands[0] = 0.0
        demands[1:] = np.random.randint(1, max_demand + 1, size=num_customers)
        
        return cls(coords, demands, capacity)


class RouteDecoder:
    """
    Classe responsável por decodificar um cromossomo (Tour Gigante) em rotas viáveis do CVRP.
    Utiliza um algoritmo guloso de particionamento (Split) baseado em restrição de capacidade.
    """
    def __init__(self, instance):
        """
        Inicializa o decodificador com a instância do problema.
        
        Args:
            instance (CVRPInstance): Instância de CVRP contendo demandas e matriz de distância.
        """
        self.instance = instance

    def decode(self, chromosome):
        """
        Decodifica o cromossomo (permutação de clientes) em uma lista de rotas viáveis.
        A lógica agrupa os clientes sequencialmente na ordem em que aparecem no cromossomo.
        Se adicionar o próximo cliente violar a capacidade máxima do veículo Q,
        a rota é fechada (retorna ao depósito) e uma nova rota é iniciada.
        
        Args:
            chromosome (list ou np.ndarray): Permutação dos índices dos clientes (de 1 a N).
            
        Returns:
            tuple: (routes, total_distance)
                - routes (list of lists): Lista contendo as rotas. Cada rota é uma lista de clientes.
                - total_distance (float): Soma das distâncias percorridas por todos os veículos.
        """
        routes = []
        current_route = []
        current_load = 0.0
        total_distance = 0.0
        
        # Iterar sobre a permutação de clientes no cromossomo
        for customer in chromosome:
            demand = self.instance.demands[customer]
            
            # Validação educativa: caso a demanda do cliente seja maior que a capacidade total do veículo
            if demand > self.instance.capacity:
                raise ValueError(f"A demanda do cliente {customer} ({demand}) excede a capacidade máxima do veículo ({self.instance.capacity}).")
            
            # Se couber na rota atual, adicionamos o cliente
            if current_load + demand <= self.instance.capacity:
                current_route.append(customer)
                current_load += demand
            else:
                # Se não couber, fechamos a rota atual e calculamos sua distância
                routes.append(current_route)
                total_distance += self._calculate_route_distance(current_route)
                
                # Iniciamos uma nova rota com o cliente atual
                current_route = [customer]
                current_load = demand
                
        # Adiciona a última rota pendente
        if current_route:
            routes.append(current_route)
            total_distance += self._calculate_route_distance(current_route)
            
        return routes, total_distance

    def _calculate_route_distance(self, route):
        """
        Calcula a distância total de uma rota específica, incluindo
        o trajeto do depósito (0) ao primeiro cliente, entre os clientes da rota,
        e do último cliente de volta ao depósito (0).
        
        Args:
            route (list): Lista de índices de clientes atendidos na rota.
            
        Returns:
            float: Distância total da rota.
        """
        if not route:
            return 0.0
            
        # Rota começa no depósito (0)
        distance = self.instance.dist_matrix[0, route[0]]
        
        # Distâncias intermediárias entre os clientes sucessivos
        for i in range(len(route) - 1):
            distance += self.instance.dist_matrix[route[i], route[i+1]]
            
        # Rota termina voltando ao depósito (0)
        distance += self.instance.dist_matrix[route[-1], 0]
        
        return distance
