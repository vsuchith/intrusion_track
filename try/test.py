import pika

VHOST = "/"                 # change if you use a custom vhost
HOST = "localhost"
EXCHANGE = "video_direct"
QUEUE = "video_frames"
RK = "frames"

def on_return(ch, method, props, body):
    print("UNROUTABLE:", method.reply_code, method.reply_text, "rk=", method.routing_key)

conn = pika.BlockingConnection(
    pika.ConnectionParameters(host=HOST, virtual_host=VHOST, heartbeat=30)
)
ch = conn.channel()

# Declare exchange/queue/bind (durable, no TTL/length caps)
ch.exchange_declare(exchange=EXCHANGE, exchange_type="direct", durable=True)
ch.queue_declare(queue=QUEUE, durable=True, arguments={})
ch.queue_bind(queue=QUEUE, exchange=EXCHANGE, routing_key=RK)

# Diagnostics: publisher confirms + mandatory return
ch.confirm_delivery()
ch.add_on_return_callback(on_return)

sent = 0
for i in range(10):
    ok = ch.basic_publish(
        exchange=EXCHANGE,
        routing_key=RK,
        body=f"frame-{i}".encode(),
        properties=pika.BasicProperties(delivery_mode=2),  # persistent
        mandatory=True,  # tells us if unroutable
    )
    sent += 1
    print("published", i, "confirmed=", ok)

print("sent:", sent)
conn.close()
