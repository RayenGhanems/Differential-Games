OPTFLAGS := -O3 -lnlopt -ffast-math -pthread
WHENWORKS := -DNDEBUG 

OUTPUT_DG := outputs_DG
OUTPUT_LQ := outputs_LQ

OBJS_DG :=$(OUTPUT_DG)/DG_Structure.o $(OUTPUT_DG)/DG_Control.o 
OBJS_LQ :=$(OUTPUT_LQ)/LQ_Structure.o $(OUTPUT_LQ)/LQ_Control.o 

all_DG: $(OUTPUT_DG) $(OBJS_DG)
	@echo "Combining Everything..."
	@g++ -Wall -std=c++11 $(OPTFLAGS) $(OBJS_DG) -o DG -lnlopt -lm
	@echo "Done ✔"

$(OUTPUT_DG)/DG_Control.o: DG_Control.cpp DG_Structure.h | $(OUTPUT_DG)
	@echo "Compiling DG_Control..."
	@g++ -c $(OPTFLAGS) DG_Control.cpp -o $(OUTPUT_DG)/DG_Control.o

$(OUTPUT_DG)/DG_Structure.o: DG_Structure.cpp DG_Structure.h | $(OUTPUT_DG)
	@echo "Compiling DG_Structure..."
	@g++ -c $(OPTFLAGS) DG_Structure.cpp -o $(OUTPUT_DG)/DG_Structure.o

$(OUTPUT_DG):
	@mkdir -p $(OUTPUT_DG)

clean_DG:
	@echo "Removing all files in outputs_DG..."
	@find $(OUTPUT_DG) -type f -delete
	@echo "Done ✔"

run_DG:
	@./DG



all_LQ: $(OUTPUT_LQ) $(OBJS_LQ)
	@echo "Combining Everything..."
	@g++ -Wall -std=c++11 $(OPTFLAGS) $(OBJS_LQ) -o LQ -lnlopt -lm
	@echo "Done ✔"

$(OUTPUT_LQ)/LQ_Control.o: LQ_Control.cpp LQ_Structure.h | $(OUTPUT_LQ)
	@echo "Compiling LQ_Control..."
	@g++ -c $(OPTFLAGS) LQ_Control.cpp -o $(OUTPUT_LQ)/LQ_Control.o

$(OUTPUT_LQ)/LQ_Structure.o: LQ_Structure.cpp LQ_Structure.h | $(OUTPUT_LQ)
	@echo "Compiling LQ_Structure..."
	@g++ -c $(OPTFLAGS) LQ_Structure.cpp -o $(OUTPUT_LQ)/LQ_Structure.o

$(OUTPUT_LQ):
	@mkdir -p $(OUTPUT_LQ)

clean_LQ:
	@echo "Removing all files in outputs_LQ..."
	@find $(OUTPUT_LQ) -type f -delete
	@echo "Done ✔"

run_LQ:
	@./LQ