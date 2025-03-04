$env:CARGO_INCREMENTAL = "0"; $env:RUSTFLAGS = "-Cinstrument-coverage"; $env:LLVM_PROFILE_FILE = "cargo-test-%p-%m.profraw"; cargo test;
grcov . --binary-path ./target/debug/deps/ -s . -t html --branch --ignore-not-existing --ignore '../*' --ignore "/*" -o target/coverage/html
Remove-Item *.profraw
Start-Process "target/coverage/html/index.html"