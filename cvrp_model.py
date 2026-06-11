import numpy as np

class CVRPInstance:

    def __init__(self, coords, demands, capacity):

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

    def __init__(self, instance):
  
        self.instance = instance

    def decode(self, chromosome):

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
