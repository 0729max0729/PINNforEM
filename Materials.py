import torch
from pina import Condition
from pina.geometry import Location
from Equations import Maxwell3DEquation, InterfaceEMFieldEquation


class Material:
    """
    Material class representing material properties for 3D electromagnetics simulation.
    """

    def __init__(self, name, epsilon, sigma, mu, location: Location):
        self.name = name
        self.epsilon = epsilon
        self.sigma = sigma
        self.mu = mu
        self.location = location

    def apply_to_equation(self, interface_material=None):
        """
        Apply material properties based on the location sample_mode.
        """
        if self.location.sample_mode == 'interior':
            return Maxwell3DEquation(sigma=self.sigma, epsilon=self.epsilon, mu=self.mu)

        elif self.location.sample_mode == 'edges' and interface_material:
            normal_vectors = self.location.calculate_normal_vector()
            edge_equations = []

            for normal_vector in normal_vectors:
                eq = InterfaceEMFieldEquation(
                    epsilon_1=self.epsilon,
                    epsilon_2=interface_material.epsilon,
                    mu_1=self.mu,
                    mu_2=interface_material.mu,
                    normal_vector=normal_vector,
                    sigma_1=self.sigma,
                    sigma_2=interface_material.sigma
                )
                edge_equations.append(eq)

            return edge_equations

        elif self.location.sample_mode == 'both' and interface_material:
            normal_vectors = self.location.calculate_normal_vector()
            edge_equations = [
                InterfaceEMFieldEquation(
                    epsilon_1=self.epsilon,
                    epsilon_2=interface_material.epsilon,
                    mu_1=self.mu,
                    mu_2=interface_material.mu,
                    normal_vector=normal_vector,
                    sigma_1=self.sigma,
                    sigma_2=interface_material.sigma
                )
                for normal_vector in normal_vectors
            ]
            return {
                'interior': Maxwell3DEquation(sigma=self.sigma, epsilon=self.epsilon, mu=self.mu),
                'edges': edge_equations
            }
        else:
            raise ValueError("Unsupported sample_mode. Choose from 'interior', 'edges', 'both'.")




class MaterialHandler:
    """
    Manage multiple materials and automatically generate interior and interface conditions in 3D space.
    """
    def __init__(self, materials):
        self.materials = materials

    def apply_equations(self):
        conditions = {}

        for material in self.materials:
            # Apply interior conditions
            if material.location.sample_mode in ['interior', 'both']:
                conditions[f'interior_{material.name}'] = Condition(
                    location=material.location,
                    equation=Maxwell3DEquation(
                        sigma=material.sigma,
                        epsilon=material.epsilon,
                        mu=material.mu
                    )
                )

            # Apply interface conditions
            if material.location.sample_mode in ['edges', 'both']:
                for neighbor in self.materials:
                    if material == neighbor:
                        continue

                    normal_vectors = material.location.calculate_normal_vector()
                    for i, normal in enumerate(normal_vectors):
                        conditions[f'interface_{material.name}_to_{neighbor.name}_edge_{i}'] = Condition(
                            location=material.location,
                            equation=InterfaceEMFieldEquation(
                                epsilon_1=material.epsilon,
                                epsilon_2=neighbor.epsilon,
                                mu_1=material.mu,
                                mu_2=neighbor.mu,
                                normal_vector=normal,
                                sigma_1=material.sigma,
                                sigma_2=neighbor.sigma
                            )
                        )

        return conditions
