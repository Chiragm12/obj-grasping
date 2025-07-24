from comprehensive_test import ComprehensiveObjectTester

def quick_test():
    """Quick test with better error handling and encoding support"""
    print("=== Quick Multi-Object Test (3 tests per object) ===")
    
    try:
        tester = ComprehensiveObjectTester() 
        
        results = tester.test_all_custom_objects(
            tests_per_object=3,
            gui=False
        )
        
        if results and 'overall_stats' in results:
            overall_rate = results['overall_stats']['overall_success_rate']
            total_objects = results['overall_stats']['total_objects']
            total_tests = results['overall_stats']['total_tests']
            
            print(f"\n" + "="*50)
            print(f"QUICK TEST RESULTS:")
            print(f"Objects Tested: {total_objects}")
            print(f"Total Tests: {total_tests}")
            print(f"Overall Success Rate: {overall_rate:.1f}%")
            print(f"="*50)
            
            return results
        else:
            print("[ERROR] No valid results generated")
            return None
            
    except UnicodeEncodeError as e:
        print(f"[ENCODING ERROR] Unicode issue: {e}")
        print("Try running with UTF-8 console encoding")
        return None
    except Exception as e:
        print(f"[GENERAL ERROR] {e}")
        return None

if __name__ == "__main__":
    quick_test()
