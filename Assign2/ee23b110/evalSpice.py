# evalSpice.py

"""
Roll No: EE23B110
Name: Ishaan Seth
Date: 08 Sept 2024
Version: 2
Description: To write a function (evalSpice) that will read the given SPICE circuit, parse it and solve the circuit by finding all node voltages and branch currents of branches with a voltage source, and raise specific errors for invalid definitions.
Inputs: filename (Name of the file containing the SPICE circuit)
Outputs: voltages (Dictionary containing all node voltages), currents (Dictionary containing branch currents through each voltage source)
"""

import numpy as np


def evalSpice(filename):
    try:
        # Open the SPICE file and read the circuit
        with open(filename, "r") as file:
            lines = file.readlines()

        circuit_start = None
        circuit_end = None

        # Finding .circuit and .end boundaries
        start_flag = False
        end_flag = False
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line == ".circuit":
                if start_flag == False:
                    circuit_start = i
                    start_flag = True
                elif start_flag == True:
                    raise ValueError("Invalid circuit definition")
            elif stripped_line == ".end":
                if end_flag == False:
                    circuit_end = i
                    end_flag = True
                elif end_flag == True:
                    raise ValueError("Invalid circuit definition")

        # Check if the file has valid .circuit and .end markers
        if circuit_start is None or circuit_end is None or circuit_start >= circuit_end:
            raise ValueError("Malformed circuit file")

        # Extract the component list between .circuit and .end markers
        component_list = lines[circuit_start + 1 : circuit_end]
        voltage_sources = []
        current_sources = []
        resistors = []
        nodes = set()

        # Parse the components and categorize them
        for line in component_list:
            tokens = line.split()
            if len(tokens) < 4:
                raise ValueError("Invalid component definition")

            component_name = tokens[0]
            node1 = tokens[1]
            node2 = tokens[2]
            if node1 == node2:
                raise ValueError("Invalid node definition")
            nodes.update([node1, node2])

            # Categorize components based on their type
            if component_name.startswith("R"):  # Resistor
                resistors.append(
                    {
                        "name": component_name,
                        "node1": node1,
                        "node2": node2,
                        "value": float(tokens[3]),
                    }
                )
            elif component_name.startswith("V"):  # Voltage Source
                if len(tokens) < 5:
                    raise ValueError("Invalid component definition")

                voltage_sources.append(
                    {
                        "name": component_name,
                        "node1": node1,  # Positive node of voltage source
                        "node2": node2,
                        "value": float(tokens[4]),
                    }
                )
            elif component_name.startswith("I"):  # Current Source
                if len(tokens) < 5:
                    raise ValueError("Invalid component definition")

                current_sources.append(
                    {
                        "name": component_name,
                        "node1": node1,
                        "node2": node2,  # Current flows out through this node
                        "value": float(tokens[4]),
                    }
                )
            else:
                raise ValueError("Only V, I, R elements are permitted")

        # Node Mapping
        node_map = {}
        node_count = 0

        # Ensure GND node is present and assign it index 0
        if "GND" in nodes:
            node_map["GND"] = 0
        else:
            raise ValueError(
                "GND node is missing from the circuit. Please include a GND node."
            )

        # Remove GND from the node set and map remaining nodes
        nodes_without_gnd = sorted(node for node in nodes if node != "GND")

        for idx, node in enumerate(nodes_without_gnd, start=1):
            node_map[node] = idx
            node_count += 1

        num_voltage_sources = len(voltage_sources)
        matrix_num = (
            node_count + num_voltage_sources
        )  # Total number of rows/columns in the matrix

        # Initialize matrices for nodal analysis
        G = np.zeros(
            (matrix_num, matrix_num)
        )  # Conductance matrix, containing conductances and coefficients for voltage source equations
        b = np.zeros(
            (matrix_num, 1)
        )  # Column vector containing voltage and current source values

        # Ensure current sources are not in series with different values
        for i in range(len(current_sources)):
            for j in range(i + 1, len(current_sources)):
                i1_node1 = current_sources[i]["node1"]
                i1_node2 = current_sources[i]["node2"]
                i2_node1 = current_sources[j]["node1"]
                i2_node2 = current_sources[j]["node2"]

                if (
                    (i1_node1 == i2_node1 and i1_node2 == i2_node2)
                    or (i1_node1 == i2_node2 and i1_node2 == i2_node1)
                ) and (current_sources[i]["value"] != current_sources[j]["value"]):
                    raise ValueError("Circuit error: no solution")

        # Ensure voltage sources are not in parallel in between the same pair of nodes with different values
        for i in range(len(voltage_sources)):
            for j in range(i + 1, len(voltage_sources)):
                v1_node1 = voltage_sources[i]["node1"]
                v1_node2 = voltage_sources[i]["node2"]
                v2_node1 = voltage_sources[j]["node1"]
                v2_node2 = voltage_sources[j]["node2"]

                if (
                    (v1_node1 == v2_node1 and v1_node2 == v2_node2)
                    or (v1_node1 == v2_node2 and v1_node2 == v2_node1)
                ) and (voltage_sources[i]["value"] != voltage_sources[j]["value"]):
                    raise ValueError("Circuit error: no solution")

        # Fill in conductances in the matrix
        for resistor in resistors:
            n1 = node_map[resistor["node1"]]
            n2 = node_map[resistor["node2"]]
            value = resistor["value"]

            if n1 != 0:  # Only update if n1 is not GND
                G[n1 - 1, n1 - 1] += 1 / value  # Self-conductance at n1

            if n2 != 0:  # Only update if n2 is not GND
                G[n2 - 1, n2 - 1] += 1 / value  # Self-conductance at n2

            if n1 != 0 and n2 != 0:  # Mutual conductance between n1 and n2
                G[n1 - 1, n2 - 1] -= 1 / value
                G[n2 - 1, n1 - 1] -= 1 / value

        # Fill in voltage sources in the matrix
        for i, vsource in enumerate(voltage_sources):
            n1 = node_map[vsource["node1"]]
            n2 = node_map[vsource["node2"]]
            row = node_count + i  # Voltage sources are added after all node equations

            if n1 != 0:  # n1 is not GND
                G[n1 - 1, row] = 1  # Voltage source adds 1 at n1
                G[row, n1 - 1] = 1  # Symmetric entry

            if n2 != 0:  # n2 is not GND
                G[n2 - 1, row] = -1  # Voltage source subtracts 1 at n2
                G[row, n2 - 1] = -1  # Symmetric entry

            b[row, 0] = vsource["value"]

        # Fill in current sources into b vector
        for current_source in current_sources:
            n1 = node_map[current_source["node1"]]
            n2 = node_map[current_source["node2"]]
            value = current_source["value"]

            if n1 != 0:
                b[n1 - 1, 0] -= value

            if n2 != 0:
                b[n2 - 1, 0] += value

        # Solve the system of equations
        try:
            x = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            raise ValueError("Circuit error: no solution")

        # Return the voltages and currents
        voltages = {}
        for node, idx in node_map.items():
            if idx != 0:
                voltages[node] = x[idx - 1, 0].item()

        voltages["GND"] = 0.0  # Set the GND node voltage

        currents = {}
        for i, vsource in enumerate(voltage_sources):
            currents[vsource["name"]] = x[node_count + i, 0].item()

        return voltages, currents

    except FileNotFoundError:
        raise FileNotFoundError("Please give the name of a valid SPICE file as input")
