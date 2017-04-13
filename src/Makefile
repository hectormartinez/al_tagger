lex2embeds:	infos	\
		trace_lex2_embeds.bg	\
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

lexembeds:	infos	\
		trace_lex_embeds.ar	\
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

lexcoarseembeds:infos	\
		trace_lexcoarse_embeds.ar	\
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

infos:
	echo "python = "`which python`
	echo "PATH = " $PATH


trace_lex2_embeds.%:
	python run_lex2_embeds.py --dynet-seed 113 --lang $* > $@ 2>&1

trace_lex_embeds.%:
	python run_lex_embeds.py --dynet-seed 113 --lang $* > $@ 2>&1

trace_lexcoarse_embeds.%:
	python run_lexcoarse_embeds.py --dynet-seed 113 --lang $* > $@ 2>&1
