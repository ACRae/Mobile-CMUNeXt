# run inside src/images
mkdir -p embedded
for f in *.bin; do
  [ -f "$f" ] || continue
  base="${f%.bin}"
  sym="$(printf 'img_%s' "$base" | tr -cs 'A-Za-z0-9' '_')"
  xxd -i -n "$sym" "$f" > "embedded/${base}.h"
done
echo "Wrote headers to embedded/"