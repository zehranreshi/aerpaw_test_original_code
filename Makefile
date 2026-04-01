.PHONY: test test-fast

test:
	./run_tests.sh

test-fast:
	./run_tests.sh --no-install
