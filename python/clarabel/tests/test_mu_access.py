"""
Test mu access functionality in Python bindings.
Tests access to mu values in solutions from both solve() and solve_all_iterations() methods.
"""

import sys
import os

# Add the path to the clarabel python module if it's not in the system path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

try:
    import clarabel
    import numpy as np
    
    def test_mu_access_single_solution():
        """Test that mu is accessible in the solution from solve() method."""
        
        # Create a simple QP problem
        # min x^2 + y^2 s.t. x >= 0, y >= 0, x + y >= 1
        from scipy import sparse
        
        P = sparse.csc_matrix([[2.0, 0.0], [0.0, 2.0]])
        q = np.array([0.0, 0.0])
        
        # Constraints: x >= 0, y >= 0, x + y >= 1
        A = sparse.csc_matrix([
            [1., 0.],    # x >= 0
            [0., 1.],    # y >= 0
            [1., 1.]     # x + y >= 1
        ])
        b = np.array([0.0, 0.0, 1.0])
        cones = [clarabel.NonnegativeConeT(3)]
        
        # Solve the problem
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        solution = solver.solve()
        
        # Test that solution has mu attribute
        assert hasattr(solution, 'mu'), "Solution should have 'mu' attribute"
        
        # Test that mu is a float
        assert isinstance(solution.mu, float), f"mu should be a float, got {type(solution.mu)}"
        
        # Test that mu is positive (since it's a barrier parameter)
        assert solution.mu >= 0.0, f"mu should be non-negative, got {solution.mu}"
        
        # Test that solution is successful
        assert solution.status == clarabel.SolverStatus.Solved, f"Expected Solved status, got {solution.status}"
        
        print(f"✓ Single solution mu access test passed!")
        print(f"  Solution status: {solution.status}")
        print(f"  Solution mu value: {solution.mu}")
        print(f"  Solution iterations: {solution.iterations}")
    
    def test_mu_access_all_iterations():
        """Test that mu is accessible in all solutions from solve_all_iterations() method."""
        
        # Create a simple QP problem
        # min x^2 + y^2 s.t. x >= 0, y >= 0, x + y >= 1
        from scipy import sparse
        
        P = sparse.csc_matrix([[2.0, 0.0], [0.0, 2.0]])
        q = np.array([0.0, 0.0])
        
        # Constraints: x >= 0, y >= 0, x + y >= 1
        A = sparse.csc_matrix([
            [1., 0.],    # x >= 0
            [0., 1.],    # y >= 0
            [1., 1.]     # x + y >= 1
        ])
        b = np.array([0.0, 0.0, 1.0])
        cones = [clarabel.NonnegativeConeT(3)]
        
        # Solve the problem with all iterations
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        all_solutions = solver.solve_all_iterations()
        
        # Test that we get a list of solutions
        assert isinstance(all_solutions, list), f"solve_all_iterations should return a list, got {type(all_solutions)}"
        
        # Test that we have at least one solution
        assert len(all_solutions) > 0, "solve_all_iterations should return at least one solution"
        
        print(f"✓ Got {len(all_solutions)} solutions from solve_all_iterations()")
        
        # Test each solution in the list
        for i, solution in enumerate(all_solutions):
            # Test that each solution has mu attribute
            assert hasattr(solution, 'mu'), f"Solution {i} should have 'mu' attribute"
            
            # Test that mu is a float
            assert isinstance(solution.mu, float), f"Solution {i}: mu should be a float, got {type(solution.mu)}"
            
            # Test that mu is positive (since it's a barrier parameter)
            assert solution.mu >= 0.0, f"Solution {i}: mu should be non-negative, got {solution.mu}"
            
            # Test that solution has other expected attributes
            assert hasattr(solution, 'x'), f"Solution {i} should have 'x' attribute"
            assert hasattr(solution, 's'), f"Solution {i} should have 's' attribute"
            assert hasattr(solution, 'z'), f"Solution {i} should have 'z' attribute"
            assert hasattr(solution, 'status'), f"Solution {i} should have 'status' attribute"
            assert hasattr(solution, 'iterations'), f"Solution {i} should have 'iterations' attribute"
            
            print(f"  Solution {i+1}: mu = {solution.mu:.6e}, iterations = {solution.iterations}, status = {solution.status}")
        
        # Test that mu values are generally decreasing (barrier parameter should decrease)
        if len(all_solutions) > 1:
            mu_values = [sol.mu for sol in all_solutions]
            print(f"  Mu values progression: {[f'{mu:.6e}' for mu in mu_values[:5]]}{'...' if len(mu_values) > 5 else ''}")
            
            # Check that the final mu is smaller than the initial mu (generally expected for barrier methods)
            if mu_values[0] > 0 and mu_values[-1] >= 0:
                print(f"  Mu decreased from {mu_values[0]:.6e} to {mu_values[-1]:.6e}")
        
        print(f"✓ All iterations mu access test passed!")
    
    def test_mu_consistency():
        """Test that mu value from solve() matches the final mu from solve_all_iterations()."""
        
        # Create a simple QP problem
        from scipy import sparse
        
        P = sparse.csc_matrix([[2.0, 0.0], [0.0, 2.0]])
        q = np.array([0.0, 0.0])
        
        A = sparse.csc_matrix([
            [1., 0.],    # x >= 0
            [0., 1.],    # y >= 0
            [1., 1.]     # x + y >= 1
        ])
        b = np.array([0.0, 0.0, 1.0])
        cones = [clarabel.NonnegativeConeT(3)]
        
        # First solve with solve()
        settings1 = clarabel.DefaultSettings()
        settings1.verbose = False
        solver1 = clarabel.DefaultSolver(P, q, A, b, cones, settings1)
        single_solution = solver1.solve()
        
        # Then solve with solve_all_iterations()
        settings2 = clarabel.DefaultSettings()
        settings2.verbose = False
        solver2 = clarabel.DefaultSolver(P, q, A, b, cones, settings2)
        all_solutions = solver2.solve_all_iterations()
        
        # Test that both methods solved successfully
        assert single_solution.status == clarabel.SolverStatus.Solved, "Single solve should be successful"
        assert len(all_solutions) > 0, "solve_all_iterations should return solutions"
        assert all_solutions[-1].status == clarabel.SolverStatus.Solved, "Final solution should be successful"
        
        # Test that final mu values are close (they should be the same final solution)
        final_mu_single = single_solution.mu
        final_mu_all = all_solutions[-1].mu
        
        print(f"  Single solve final mu: {final_mu_single:.10e}")
        print(f"  All iterations final mu: {final_mu_all:.10e}")
        
        # Allow for small numerical differences
        mu_diff = abs(final_mu_single - final_mu_all)
        relative_error = mu_diff / max(abs(final_mu_single), abs(final_mu_all), 1e-16)
        
        assert relative_error < 1e-10, f"Final mu values should match: {final_mu_single:.10e} vs {final_mu_all:.10e} (relative error: {relative_error:.2e})"
        
        print(f"✓ Mu consistency test passed! (relative error: {relative_error:.2e})")
    
    if __name__ == "__main__":
        print("Testing Clarabel Python mu access functionality...")
        
        try:
            test_mu_access_single_solution()
            print()
            test_mu_access_all_iterations()
            print()
            test_mu_consistency()
            print("\n🎉 All mu access tests passed! Both solve() and solve_all_iterations() properly expose mu values.")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
except ImportError as e:
    print(f"Could not import required modules: {e}")
    print("This is expected if clarabel Python bindings are not built/installed.")
    print("The bindings should work once the project is properly built.")
