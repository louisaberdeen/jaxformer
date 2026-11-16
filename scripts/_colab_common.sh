# Shared plumbing for the Colab driver scripts (train_colab.sh, profile_colab.sh).
# Not runnable on its own — `source` it after setting $ROOT (repo root) and $S (session).

# Print how to stop the VM. Wired as an EXIT trap by colab_provision so it shows on
# success AND on error/Ctrl-C. It only PRINTS — the VM is intentionally left running so
# you can re-run a driver against the same session without re-provisioning.
# ponytail: no auto-stop — an idle VM keeps billing until you run the printed command.
_colab_stop_reminder() {
  echo
  echo "[driver] VM '$S' left running. Stop it (billing!) with:"
  echo "    colab stop -s $S"
}

colab_build_wheel() {            # sets $WHEEL
  echo "[driver] building wheel"
  rm -f "$ROOT"/dist/*.whl
  ( cd "$ROOT" && uv build --wheel )
  WHEEL="$(ls -t "$ROOT"/dist/*.whl | head -1)"
}

colab_provision() {              # $1 = accel flags, e.g. "--tpu v6e1" (may be empty)
  # Reuse the VM if it's already up (the drivers leave it running); only provision when
  # there's no live session by this name. Idempotent so you can re-run to iterate.
  if colab status -s "$S" 2>&1 | grep -q 'Status:'; then
    echo "[driver] reusing existing session '$S'"
  else
    echo "[driver] provisioning session '$S' (${1:-cpu})"
    colab --auth=adc new -s "$S" $1
  fi
  trap _colab_stop_reminder EXIT   # VM is billable from here on
}

colab_install_wheel() {          # upload + install wheel (--no-deps) + light deps
  # NB: `colab upload` needs a full dest FILE path — a dir like /content/ 500s (CLI 0.6.0).
  local whl="/content/$(basename "$WHEEL")"
  colab upload -s "$S" "$WHEEL" "$whl"
  echo "[driver] installing jaxformer (--no-deps keeps the VM's preinstalled jax[tpu])"
  # `colab install` has no --no-deps flag (0.6.0), so pip the wheel directly via exec;
  # --force-reinstall so a re-run picks up the freshly built same-version wheel.
  echo "import sys,subprocess; subprocess.run([sys.executable,'-m','pip','install','--no-deps','--force-reinstall','$whl'],check=True)" | colab exec -s "$S"
  colab install -s "$S" tokenizers mlflow huggingface_hub tqdm pyarrow pandas
}

colab_pin_jax() {                # align the VM's jax to the version the code targets
  # The VM ships jaxlib 0.7.2, whose Mosaic frontend emits IR v8, but its libtpu only
  # deserializes Mosaic <= 7 — so the flash-attention Pallas kernel fails to compile
  # ("Unsupported version: expected <= 7 but got 8"). The repo pins jax<0.7 (verified on
  # 0.6.2), whose jaxlib emits Mosaic <= 7 and ships a matching libtpu. Idempotent: skip
  # when already pinned so reused VMs don't reinstall every run.
  local want="${PIN_JAX:-0.6.2}"
  if echo "import jax; print('JAX_'+jax.__version__)" | colab exec -s "$S" --timeout 60 2>/dev/null | grep -q "JAX_$want"; then
    echo "[driver] jax already $want on the VM"
    return
  fi
  echo "[driver] pinning jax[tpu]==$want on the VM (fixes Mosaic v8 vs libtpu v7)"
  echo "import sys,subprocess; subprocess.run([sys.executable,'-m','pip','install','--force-reinstall','jax[tpu]==$want','-f','https://storage.googleapis.com/jax-releases/libtpu_releases.html'],check=True)" \
    | colab exec -s "$S" --timeout 900
  colab restart-kernel -s "$S"   # reload the new jaxlib/libtpu (wheel stays installed on disk)
  # Verify in a SUBPROCESS, not the kernel: jax.devices() grabs the single TPU for the life
  # of the calling process, and train_colab.sh launches training as a *separate* detached
  # process — if the kernel held the TPU, that child couldn't initialize it.
  echo "import sys,subprocess; print(subprocess.run([sys.executable,'-c','import jax;print(\"[driver] jax\",jax.__version__,jax.devices())'],capture_output=True,text=True).stdout,end='')" \
    | colab exec -s "$S" --timeout 180
}
