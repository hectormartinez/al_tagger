lex2embeds:	trace_lex2_embeds.bg	\
		trace_lex2_embeds.cs	\
		trace_lex2_embeds.da	\
		trace_lex2_embeds.de	\
		trace_lex2_embeds.el	\
		trace_lex2_embeds.en	\
		trace_lex2_embeds.es	\
		trace_lex2_embeds.et	\
		trace_lex2_embeds.fa	\
		trace_lex2_embeds.fr	\
		trace_lex2_embeds.ga	\
		trace_lex2_embeds.hr	\
		trace_lex2_embeds.id	\
		trace_lex2_embeds.it	\
		trace_lex2_embeds.nl	\
		trace_lex2_embeds.no	\
		trace_lex2_embeds.pl	\
		trace_lex2_embeds.pt	\
		trace_lex2_embeds.sl	\
		trace_lex2_embeds.sv

lexembeds:	trace_lex_embeds.ar	\
		trace_lex_embeds.bg	\
		trace_lex_embeds.ca	\
		trace_lex_embeds.cs	\
		trace_lex_embeds.da	\
		trace_lex_embeds.de	\
		trace_lex_embeds.el	\
		trace_lex_embeds.en	\
		trace_lex_embeds.es	\
		trace_lex_embeds.et	\
		trace_lex_embeds.eu	\
		trace_lex_embeds.fi	\
		trace_lex_embeds.fr	\
		trace_lex_embeds.he	\
		trace_lex_embeds.hi	\
		trace_lex_embeds.id	\
		trace_lex_embeds.it	\
		trace_lex_embeds.nl	\
		trace_lex_embeds.no	\
		trace_lex_embeds.pl	\
		trace_lex_embeds.pt	\
		trace_lex_embeds.sl	\
		trace_lex_embeds.sv

lexcoarseembeds:trace_lexcoarse_embeds.ar	\
		trace_lexcoarse_embeds.bg	\
		trace_lexcoarse_embeds.ca	\
		trace_lexcoarse_embeds.cs	\
		trace_lexcoarse_embeds.da	\
		trace_lexcoarse_embeds.de	\
		trace_lexcoarse_embeds.el	\
		trace_lexcoarse_embeds.en	\
		trace_lexcoarse_embeds.es	\
		trace_lexcoarse_embeds.et	\
		trace_lexcoarse_embeds.eu	\
		trace_lexcoarse_embeds.fi	\
		trace_lexcoarse_embeds.fr	\
		trace_lexcoarse_embeds.he	\
		trace_lexcoarse_embeds.hi	\
		trace_lexcoarse_embeds.id	\
		trace_lexcoarse_embeds.it	\
		trace_lexcoarse_embeds.nl	\
		trace_lexcoarse_embeds.no	\
		trace_lexcoarse_embeds.pl	\
		trace_lexcoarse_embeds.pt	\
		trace_lexcoarse_embeds.sl	\
		trace_lexcoarse_embeds.sv

lex2:		trace_lex2.bg	\
		trace_lex2.cs	\
		trace_lex2.da	\
		trace_lex2.de	\
		trace_lex2.el	\
		trace_lex2.en	\
		trace_lex2.es	\
		trace_lex2.et	\
		trace_lex2.fa	\
		trace_lex2.fr	\
		trace_lex2.ga	\
		trace_lex2.hr	\
		trace_lex2.id	\
		trace_lex2.it	\
		trace_lex2.nl	\
		trace_lex2.no	\
		trace_lex2.pl	\
		trace_lex2.pt	\
		trace_lex2.sl	\
		trace_lex2.sv

lex:		trace_lex.ar	\
		trace_lex.bg	\
		trace_lex.ca	\
		trace_lex.cs	\
		trace_lex.da	\
		trace_lex.de	\
		trace_lex.el	\
		trace_lex.en	\
		trace_lex.es	\
		trace_lex.et	\
		trace_lex.eu	\
		trace_lex.fi	\
		trace_lex.fr	\
		trace_lex.he	\
		trace_lex.hi	\
		trace_lex.id	\
		trace_lex.it	\
		trace_lex.nl	\
		trace_lex.no	\
		trace_lex.pl	\
		trace_lex.pt	\
		trace_lex.sl	\
		trace_lex.sv

lexcoarse:	trace_lexcoarse.ar	\
		trace_lexcoarse.bg	\
		trace_lexcoarse.ca	\
		trace_lexcoarse.cs	\
		trace_lexcoarse.da	\
		trace_lexcoarse.de	\
		trace_lexcoarse.el	\
		trace_lexcoarse.en	\
		trace_lexcoarse.es	\
		trace_lexcoarse.et	\
		trace_lexcoarse.eu	\
		trace_lexcoarse.fi	\
		trace_lexcoarse.fr	\
		trace_lexcoarse.he	\
		trace_lexcoarse.hi	\
		trace_lexcoarse.id	\
		trace_lexcoarse.it	\
		trace_lexcoarse.nl	\
		trace_lexcoarse.no	\
		trace_lexcoarse.pl	\
		trace_lexcoarse.pt	\
		trace_lexcoarse.sl	\
		trace_lexcoarse.sv

infos:
	echo "python = "`which python` > infos.$$$$;\
	echo "PATH = " ${PATH} >> infos.$$$$


trace_lex2_embeds.%:
	python run_lex2_embeds.py --dynet-seed 113 --lang $* > $@ 2>&1

trace_lex_embeds.%:
	python run_lex_embeds.py --dynet-seed 113 --lang $* > $@ 2>&1

trace_lexcoarse_embeds.%:
	python run_lexcoarse_embeds.py --dynet-seed 113 --lang $* > $@ 2>&1

trace_lex2.%:
	python run_simple_lex2.py --dynet-seed 113 --lang $* > $@ 2>&1

trace_lex.%:
	python run_simple_lex.py --dynet-seed 113 --lang $* > $@ 2>&1

trace_lexcoarse.%:
	python run_simple_lexcoarse.py --dynet-seed 113 --lang $* > $@ 2>&1
