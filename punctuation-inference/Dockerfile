# Dockerfile that builds the punctuation-inference executable in a musl environment, and then copies it to a scratch container
# to reduce the size of the final image.

FROM clux/muslrust:latest AS builder

RUN rustup target add x86_64-unknown-linux-musl

WORKDIR /app

COPY Docker-Cargo.toml ./Cargo.toml
COPY src/ src/

RUN cargo build --release --target x86_64-unknown-linux-musl

FROM scratch

COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/punctuation-inference /app/punctuation-inference

ENTRYPOINT ["/app/punctuation-inference"]
