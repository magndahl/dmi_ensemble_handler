import cProfile
import pstats

cProfile.run(open('test_ModelHolder.py', 'rb'), 'output_stats')

p = pstats.Stats('output_stats')
p.sort_stats('cumulative').print_stats(50)
