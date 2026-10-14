//go:build darwin

package metal

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"os"
	"testing"
	"unsafe"

	"github.com/egoist/mygo/internal/gpu"
	"github.com/egoist/mygo/internal/gpu/gputest"
	"github.com/egoist/mygo/internal/scene"
)

// unavailable ends a test for which Metal cannot be used: it skips it, or, where
// MYGO_GPU_REQUIRED is set (a machine with a GPU, CI for it), fails it, so that
// a renderer that does not start does not turn the identity tests into passing
// skips.
func unavailable(t testing.TB, err error) {
	t.Helper()
	if os.Getenv("MYGO_GPU_REQUIRED") != "" {
		t.Fatal("no Metal (MYGO_GPU_REQUIRED is set):", err)
	}
	t.Skip("no Metal:", err)
}

func TestDrawsAsTheCPURenderer(t *testing.T) {
	r, err := newRenderer()
	if err != nil {
		unavailable(t, err)
	}
	defer r.Release()
	s := gputest.Scene()
	// Twice: the second frame updates what the first uploaded.
	for range 2 {
		pix, err := r.renderOffscreen(s)
		if err != nil {
			t.Fatal(err)
		}
		gputest.Compare(t, "metal", pix, s.Width*4, s)
	}

	for frame := range 3 {
		transformed := gputest.TransformedScene()
		if frame == 2 {
			transformed.Ops[0].Transform = scene.Translation(40, 40)
		}
		pix, err := r.renderOffscreen(transformed)
		if err != nil {
			t.Fatal(err)
		}
		gputest.Compare(t, "metal-affine", pix, transformed.Width*4, transformed)
	}
	for range 2 {
		s = gputest.PerspectiveScene()
		pix, err := r.renderOffscreen(s)
		if err != nil {
			t.Fatal(err)
		}
		gputest.Compare(t, "metal-perspective", pix, s.Width*4, s)
	}
	s = gputest.ContinuousScene()
	pix, err := r.renderOffscreen(s)
	if err != nil {
		t.Fatal(err)
	}
	gputest.Compare(t, "metal-continuous", pix, s.Width*4, s)
}

// The CPU presenter starts without drawing resources; the first GPU
// frame creates them, and a frame after they are freed draws correctly.
func TestDrawingResourcesOnDemand(t *testing.T) {
	r, err := newRenderer()
	if err != nil {
		unavailable(t, err)
	}
	defer r.Release()
	if r.queue != 0 || r.sampler != 0 || r.empty != 0 || r.cur.pipeline != 0 {
		t.Fatal("created GPU drawing resources before a GPU frame")
	}
	s := gputest.Scene()
	for range 2 {
		pix, err := r.renderOffscreen(s)
		if err != nil {
			t.Fatal(err)
		}
		gputest.Compare(t, "metal-on-demand", pix, s.Width*4, s)
		pool(r.releaseTextures)
	}
}

// TestShaderLibrary checks that the library compiled ahead of time comes
// from shader.metal as it is, and that Metal loads it.
func TestShaderLibrary(t *testing.T) {
	if gpu.SourceSum(shaderSource) != shaderLibrarySum {
		t.Fatal("shader.metal changed since shaderlib.go was generated: run go generate ./internal/gpu/metal on macOS")
	}
	r, err := newRenderer()
	if err != nil {
		unavailable(t, err)
	}
	defer r.Release()
	var lib id
	pool(func() { lib = r.compiledLibrary() })
	if lib == 0 {
		t.Fatal("Metal does not load the compiled library")
	}
	pool(func() { release(&lib) })
}

// TestChangedSince checks what frames drawn in memory copy into the
// drawables, which take turns, and after the GPU drew into one.
// TestWideColors draws wide colors into a float16 target, which keeps what
// leaves 0 to 1, and an Oklab gradient between them, which does not clamp;
// and the same scene into a BGRA8 one, which draws the nearest sRGB colors.
func TestWideColors(t *testing.T) {
	r, err := newRenderer()
	if err != nil {
		unavailable(t, err)
	}
	defer r.Release()
	if !r.SetWide(true) {
		t.Fatal(r.wideErr)
	}
	s := &scene.Scene{}
	s.Reset(32, 16, scene.Color{})
	green, red := [4]float32{-0.25, 1.1, -0.1, 1}, [4]float32{1.15, -0.2, -0.1, 1}
	s.Wide = []scene.WideColors{{Color: green, Set: scene.WideColor}, {Color: green, Color2: red, Set: scene.WideColor | scene.WideColor2}}
	srgbGreen, srgbRed := scene.Color{G: 255, A: 255}, scene.Color{R: 255, A: 255}
	s.Ops = append(s.Ops,
		scene.Op{Kind: scene.OpFill, Rect: scene.Rect{W: 32, H: 8}, Color: srgbGreen, Wide: 1},
		scene.Op{Kind: scene.OpFill, Rect: scene.Rect{Y: 8, W: 32, H: 8}, Paint: scene.PaintOklab, Gradient: [4]float32{0, 0, 32, 0},
			Color: srgbGreen, Color2: srgbRed, Wide: 2})
	// General transforms must preserve the float16 wide-gamut path.
	for i := range s.Ops {
		s.Ops[i].Transform = scene.Scaling(1.1, 1)
	}
	var px [2][4]float32
	pool(func() {
		r.waitLast()
		r.cur = &r.formats[1]
		tex := r.newTexture(s.Width, s.Height, pixelFormatRGBA16Float, usageRenderTarget|usageShaderRead, nil, 0)
		if tex == 0 {
			err = errors.New("no target")
			return
		}
		defer release(&tex)
		var cb id
		if cb, err = r.encode(s, tex); err != nil {
			return
		}
		blit := send(cb, "blitCommandEncoder")
		send(blit, "synchronizeResource:", tex)
		send(blit, "endEncoding")
		send(cb, "commit")
		send(cb, "waitUntilCompleted")
		pix := make([]uint16, s.Width*s.Height*4)
		msgGetBytes(tex, sel("getBytes:bytesPerRow:fromRegion:mipmapLevel:"), unsafe.Pointer(&pix[0]), uint(s.Width*8),
			mtlRegion{W: uint(s.Width), H: uint(s.Height), D: 1}, 0)
		at := func(x, y int) (c [4]float32) {
			for i := range c {
				c[i] = half(pix[(y*s.Width+x)*4+i])
			}
			return c
		}
		px[0], px[1] = at(4, 2), at(16, 12)
	})
	if err != nil {
		t.Fatal(err)
	}
	// The pixel holds the straight color premultiplied by alpha 1, in the
	// order of the target's components (RGBA).
	for i, want := range green {
		if d := px[0][i] - want; d > 0.01 || d < -0.01 {
			t.Errorf("solid wide color: component %d is %v, want %v", i, px[0][i], want)
		}
	}
	// Midway between two colors, at least one component is out of 0 to 1:
	// the mix of the greens and reds of the ends, in Oklab, keeps the
	// chroma the ends have beyond sRGB.
	if m := px[1]; !(m[0] < 0 || m[0] > 1 || m[1] < 0 || m[1] > 1 || m[2] < 0 || m[2] > 1) {
		t.Errorf("gradient midpoint %v is inside sRGB", m)
	}

	// A BGRA8 target draws the sRGB colors of the ops.
	pix, err := r.renderOffscreen(s)
	if err != nil {
		t.Fatal(err)
	}
	if b, g, red, a := pix[(2*s.Width+4)*4], pix[(2*s.Width+4)*4+1], pix[(2*s.Width+4)*4+2], pix[(2*s.Width+4)*4+3]; b != 0 || g != 255 || red != 0 || a != 255 {
		t.Errorf("BGRA8 target: %v %v %v %v, want the sRGB green", b, g, red, a)
	}
}

// half converts an IEEE half-precision float.
func half(h uint16) float32 {
	sign := float32(1)
	if h&0x8000 != 0 {
		sign = -1
	}
	e, m := int(h>>10)&0x1f, float32(h&0x3ff)
	switch e {
	case 0:
		return sign * m / 1024 / 16384
	case 31:
		return sign * float32(math.Inf(1))
	}
	return sign * (1 + m/1024) * float32(math.Pow(2, float64(e-15)))
}

// TestShadowSplitDrawsTheSamePixels draws panels with stacks of shadows of
// several radii, blurs, spreads and offsets, with the shadows drawn around the
// box casting them and whole, and requires the same bytes.
func TestShadowSplitDrawsTheSamePixels(t *testing.T) {
	r, err := newRenderer()
	if err != nil {
		unavailable(t, err)
	}
	defer r.Release()
	for i, c := range gputest.ShadowCases() {
		s := gputest.ShadowPanelScene(c)
		r.b.NoShadowSplit = false
		split, err := r.renderOffscreen(s)
		if err != nil {
			t.Fatal(err)
		}
		instances := len(r.b.Instances)
		r.b.NoShadowSplit = true
		whole, err := r.renderOffscreen(s)
		if err != nil {
			t.Fatal(err)
		}
		if instances <= len(r.b.Instances) {
			t.Errorf("case %d: the split drew %d instances, whole %d", i, instances, len(r.b.Instances))
		}
		if !bytes.Equal(split, whole) {
			n := 0
			for k := range split {
				if split[k] != whole[k] {
					n++
				}
			}
			t.Errorf("case %d: %d bytes differ between split and whole shadows", i, n)
		}
		gputest.Compare(t, "metal-shadow-split", split, s.Width*4, s)
	}
}

// renderRaw draws s into a target of the renderer's format, BGRA8 or, when
// wide, float16, and returns its bytes as the GPU keeps them.
func renderRaw(t *testing.T, r *Renderer, s *scene.Scene, wide bool) (raw []byte) {
	t.Helper()
	var err error
	pool(func() {
		r.waitLast()
		r.cur = &r.formats[0]
		format, bpp := uint(pixelFormatBGRA8Unorm), 4
		if wide {
			r.cur, format, bpp = &r.formats[1], pixelFormatRGBA16Float, 8
		}
		tex := r.newTexture(s.Width, s.Height, format, usageRenderTarget|usageShaderRead, nil, 0)
		if tex == 0 {
			err = errors.New("no target")
			return
		}
		defer release(&tex)
		var cb id
		if cb, err = r.encode(s, tex); err != nil {
			return
		}
		blit := send(cb, "blitCommandEncoder")
		send(blit, "synchronizeResource:", tex)
		send(blit, "endEncoding")
		send(cb, "commit")
		send(cb, "waitUntilCompleted")
		if int(send(cb, "status")) == statusError {
			err = fmt.Errorf("the frame failed: %s", describe(send(cb, "error")))
			return
		}
		raw = make([]byte, s.Width*s.Height*bpp)
		msgGetBytes(tex, sel("getBytes:bytesPerRow:fromRegion:mipmapLevel:"), unsafe.Pointer(&raw[0]), uint(s.Width*bpp),
			mtlRegion{W: uint(s.Width), H: uint(s.Height), D: 1}, 0)
	})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func differing(a, b []byte) (n int) {
	for i := range a {
		if a[i] != b[i] {
			n++
		}
	}
	return n
}

// TestCullDrawsTheSamePixels draws scenes of overlapping fills with the ops
// that later opaque fills cover left out and with all of them, in BGRA8 and
// float16 targets, and requires the same bytes, and the CPU renderer's pixels.
func TestCullDrawsTheSamePixels(t *testing.T) {
	r, err := newRenderer()
	if err != nil {
		unavailable(t, err)
	}
	defer r.Release()
	cases := append(gputest.CullCases(), gputest.CullCase{Name: "reference scene", Scene: gputest.Scene(), Culled: -1})
	for _, wide := range []bool{false, true} {
		if wide && !r.SetWide(true) {
			t.Log("no wide target:", r.wideErr)
			continue
		}
		for _, c := range cases {
			r.b.NoCull = false
			culled := renderRaw(t, r, c.Scene, wide)
			n, instances := r.b.Culled, len(r.b.Instances)
			r.b.NoCull = true
			whole := renderRaw(t, r, c.Scene, wide)
			if c.Culled >= 0 && n != c.Culled {
				t.Errorf("%s wide=%v: %d ops left out, want %d", c.Name, wide, n, c.Culled)
			}
			if n > 0 && instances >= len(r.b.Instances) {
				t.Errorf("%s wide=%v: %d instances with the cull, %d without", c.Name, wide, instances, len(r.b.Instances))
			}
			if d := differing(culled, whole); d != 0 {
				t.Errorf("%s wide=%v: %d bytes differ between culled and whole", c.Name, wide, d)
			}
			if !wide {
				gputest.Compare(t, "metal-cull-"+c.Name, culled, c.Scene.Width*4, c.Scene)
			}
		}
	}
	r.b.NoCull = false
}

// TestRingSplitDrawsTheSamePixels draws border-only fills around the core
// inside their border and whole, in BGRA8 and float16 targets, and requires
// the same bytes, and the CPU renderer's pixels.
func TestRingSplitDrawsTheSamePixels(t *testing.T) {
	r, err := newRenderer()
	if err != nil {
		unavailable(t, err)
	}
	defer r.Release()
	for _, wide := range []bool{false, true} {
		if wide && !r.SetWide(true) {
			t.Log("no wide target:", r.wideErr)
			continue
		}
		for _, c := range gputest.RingCases() {
			r.b.NoRingSplit = false
			split := renderRaw(t, r, c.Scene, wide)
			instances := len(r.b.Instances)
			r.b.NoRingSplit = true
			whole := renderRaw(t, r, c.Scene, wide)
			if c.Split && instances <= len(r.b.Instances) {
				t.Errorf("%s wide=%v: the split drew %d instances, whole %d", c.Name, wide, instances, len(r.b.Instances))
			}
			if d := differing(split, whole); d != 0 {
				t.Errorf("%s wide=%v: %d bytes differ between split and whole", c.Name, wide, d)
			}
			if !wide {
				gputest.Compare(t, "metal-ring-"+c.Name, split, c.Scene.Width*4, c.Scene)
			}
		}
	}
	r.b.NoRingSplit = false
}

// TestOpaqueCoreDrawsTheSamePixels draws fills as opaque cores, drawn without
// blending, with blended edges around them, and whole and blended, in BGRA8
// and float16 targets, with and without culling, and requires the same bytes
// (and the CPU renderer's pixels).
func TestOpaqueCoreDrawsTheSamePixels(t *testing.T) {
	r, err := newRenderer()
	if err != nil {
		unavailable(t, err)
	}
	defer r.Release()
	cases := identityCases()
	cores := 0
	for _, wide := range []bool{false, true} {
		if wide && !r.SetWide(true) {
			t.Log("no wide target:", r.wideErr)
			continue
		}
		for _, noCull := range []bool{false, true} {
			r.b.NoCull = noCull
			for _, c := range cases {
				r.b.Opaque = true
				core := renderRaw(t, r, c.Scene, wide)
				n, opaque := r.b.OpaqueCores, 0
				for _, bt := range r.b.Batches {
					if bt.Opaque {
						opaque++
					}
				}
				r.b.Opaque = false
				blended := renderRaw(t, r, c.Scene, wide)
				if n > 0 && opaque == 0 {
					t.Errorf("%s: %d cores, no opaque batch", c.Name, n)
				}
				cores += n
				if d := differing(core, blended); d != 0 {
					t.Errorf("%s wide=%v noCull=%v: %d bytes differ between opaque cores and blended fills", c.Name, wide, noCull, d)
				}
				if !wide && !noCull && c.Scene.Width*c.Scene.Height < 1e6 {
					gputest.Compare(t, "metal-opaque-"+c.Name, core, c.Scene.Width*4, c.Scene)
				}
			}
		}
	}
	if cores == 0 {
		t.Error("no fill was drawn as an opaque core")
	}
	r.b.NoCull, r.b.Opaque = false, true
}

// identityCases are the scenes of the identity tests of changes to how
// scenes are drawn: the cases of the cull, ring and shadow tests, the
// scenes of fills and opaque fills at window sizes, the reference scene and
// the scenes of transformed content.
func identityCases() []gputest.CullCase {
	cases := []gputest.CullCase{{Name: "opaque scene", Scene: gputest.OpaqueScene(900, 700)},
		{Name: "window opaque scene", Scene: gputest.OpaqueScene(3456, 2000)},
		{Name: "odd opaque scene", Scene: gputest.OpaqueScene(2049, 1031)},
		{Name: "window fills", Scene: gputest.FillScene(3456, 2000)}, {Name: "odd window fills", Scene: gputest.FillScene(2049, 1031)},
		{Name: "reference scene", Scene: gputest.Scene()}, {Name: "transformed scene", Scene: gputest.TransformedScene()},
		{Name: "perspective scene", Scene: gputest.PerspectiveScene()}, {Name: "axis clip scene", Scene: gputest.AxisClipScene()},
		{Name: "continuous scene", Scene: gputest.ContinuousScene()}}
	cases = append(cases, gputest.CullCases()...)
	for _, c := range gputest.RingCases() {
		cases = append(cases, gputest.CullCase{Name: "ring " + c.Name, Scene: c.Scene})
	}
	for i, c := range gputest.ShadowCases() {
		cases = append(cases, gputest.CullCase{Name: fmt.Sprint("shadow case ", i), Scene: gputest.ShadowPanelScene(c)})
	}
	cases = append(cases, gputest.CullCase{Name: "wide colors", Scene: gputest.WideScene()})
	for _, c := range gputest.MaskedCases() {
		cases = append(cases, gputest.CullCase{Name: c.Name, Scene: c.Scene})
	}
	// Every case again with a transformed op in front, which gives rounded
	// clips masks (the splits must not touch the clip's mask area).
	n := len(cases)
	for _, c := range cases[:n] {
		cases = append(cases, gputest.CullCase{Name: c.Name + " (transformed scene)", Scene: gputest.Transformed(c.Scene)})
	}
	return cases
}

// TestSplitsDrawTheSamePixels draws every identity case with the shadow and
// ring splits and without them, in BGRA8 and float16, and requires the same
// bytes. The cases include rounded, continuous and rotated clips with masks,
// where the pieces must not change the clip's mask area.
func TestSplitsDrawTheSamePixels(t *testing.T) {
	r, err := newRenderer()
	if err != nil {
		unavailable(t, err)
	}
	defer r.Release()
	pieces := 0
	for _, wide := range []bool{false, true} {
		if wide && !r.SetWide(true) {
			t.Log("no wide target:", r.wideErr)
			continue
		}
		for _, c := range identityCases() {
			r.b.NoRingSplit, r.b.NoShadowSplit = false, false
			split := renderRaw(t, r, c.Scene, wide)
			n := len(r.b.Instances)
			r.b.NoRingSplit, r.b.NoShadowSplit = true, true
			whole := renderRaw(t, r, c.Scene, wide)
			pieces += n - len(r.b.Instances)
			if d := differing(split, whole); d != 0 {
				t.Errorf("%s wide=%v: %d bytes differ between split and whole", c.Name, wide, d)
			}
		}
	}
	if pieces == 0 {
		t.Error("nothing was split")
	}
	r.b.NoRingSplit, r.b.NoShadowSplit = false, false
}

// TestCullTransformedShadows draws a shadow under a transform, scaled, rotated,
// scaled unevenly and in perspective, with an opaque fill over exactly what the
// planner took to be the shadow's reach before the blur was grown in the shadow's
// own units (the blur's reach, in device pixels, grows with the scale), and
// requires that culling changes no pixel; and that a cover of the whole frame
// still culls it.
func TestCullTransformedShadows(t *testing.T) {
	r, err := newRenderer()
	if err != nil {
		unavailable(t, err)
	}
	defer r.Release()
	cases := []struct {
		name string
		m    scene.Affine
		rect scene.Rect
		blur float32
		cast bool
	}{
		{"scale4", scene.Scaling(4, 4), scene.Rect{X: 50, Y: 50, W: 40, H: 40}, 20, false},
		{"scale4cast", scene.Scaling(4, 4), scene.Rect{X: 50, Y: 50, W: 40, H: 40}, 20, true},
		{"nonuniform", scene.Scaling(4, 1.5), scene.Rect{X: 50, Y: 150, W: 40, H: 40}, 20, false},
		{"rotate30x3", scene.Translation(320, 320).Mul(scene.Rotation(30)).Mul(scene.Scaling(3, 3)).Mul(scene.Translation(-20, -20)), scene.Rect{W: 40, H: 40}, 16, false},
		{"scale1.5", scene.Scaling(1.5, 1.5), scene.Rect{X: 150, Y: 150, W: 60, H: 60}, 60, false},
		{"perspective", scene.Projective(2, 0, 0, 2, 100, 100, 0.0008, 0), scene.Rect{X: 50, Y: 50, W: 60, H: 60}, 20, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			b := c.m.Bounds(c.rect)
			g := 2*c.blur + 3 + 1 + 2 // the growth the planner had, plus the occluder's inset
			x0, y0 := float32(math.Floor(float64(b.X-g))), float32(math.Floor(float64(b.Y-g)))
			x1, y1 := float32(math.Ceil(float64(b.X+b.W+g))), float32(math.Ceil(float64(b.Y+b.H+g)))
			sh := scene.Op{Kind: scene.OpShadow, Rect: c.rect, Color: scene.Color{A: 255}, Blur: c.blur, Transform: c.m}
			if c.cast {
				sh.Cast = scene.Rect{X: c.rect.X + 4, Y: c.rect.Y + 4, W: c.rect.W - 8, H: c.rect.H - 8}
			}
			s := &scene.Scene{Width: 640, Height: 640, Clear: scene.Color{R: 255, G: 255, B: 255, A: 255}}
			s.Ops = []scene.Op{sh, {Kind: scene.OpFill, Rect: scene.Rect{X: x0, Y: y0, W: x1 - x0, H: y1 - y0}, Color: scene.Color{R: 180, G: 180, B: 180, A: 255}}}
			r.b.NoCull = false
			got := renderRaw(t, r, s, false)
			culled := r.b.Culled
			r.b.NoCull = true
			want := renderRaw(t, r, s, false)
			r.b.NoCull = false
			if d := differing(got, want); d != 0 {
				t.Errorf("culled %d op(s), %d bytes differ", culled, d)
			}
		})
	}
	s := &scene.Scene{Width: 640, Height: 640}
	s.Ops = []scene.Op{
		{Kind: scene.OpShadow, Rect: scene.Rect{X: 50, Y: 50, W: 40, H: 40}, Color: scene.Color{A: 255}, Blur: 20, Transform: scene.Scaling(4, 4)},
		{Kind: scene.OpFill, Rect: scene.Rect{X: 0, Y: 0, W: 640, H: 640}, Color: scene.Color{R: 180, G: 180, B: 180, A: 255}},
	}
	renderRaw(t, r, s, false)
	if r.b.Culled != 1 {
		t.Errorf("a full-frame cover culled %d ops, want 1", r.b.Culled)
	}
}

// TestFastFillDrawsTheSamePixels draws scenes of plain fills along the
// shader's fast path (and with the clip's coverage skipped where the builder
// found it to be 1) and along its general path, in BGRA8 and float16
// targets, at the size of a window of a retina display too, and requires the
// same bytes.
func TestFastFillDrawsTheSamePixels(t *testing.T) {
	r, err := newRenderer()
	if err != nil {
		unavailable(t, err)
	}
	defer r.Release()
	cases := []gputest.CullCase{{Name: "reference scene", Scene: gputest.Scene()}, {Name: "small fills", Scene: gputest.FillScene(900, 700)},
		{Name: "window fills", Scene: gputest.FillScene(3456, 2000)}, {Name: "odd window fills", Scene: gputest.FillScene(2049, 1031)}}
	for _, c := range gputest.CullCases() {
		cases = append(cases, c)
	}
	for _, wide := range []bool{false, true} {
		if wide && !r.SetWide(true) {
			t.Log("no wide target:", r.wideErr)
			continue
		}
		for _, c := range cases {
			r.b.GeneralFill, r.b.NoClipSkip = false, false
			fast := renderRaw(t, r, c.Scene, wide)
			plain, unclipped := 0, 0
			for _, in := range r.b.Instances {
				if in.Params[0] == 0 && in.Params[2] == 0 && in.UV == [4]float32{} {
					plain++
				}
				if in.Transform1[3] == 3 {
					unclipped++
				}
			}
			r.b.GeneralFill, r.b.NoClipSkip = true, true
			general := renderRaw(t, r, c.Scene, wide)
			if c.Name == "window fills" && (plain == 0 || unclipped == 0) {
				t.Errorf("%s: %d plain fills, %d without clip coverage", c.Name, plain, unclipped)
			}
			for _, in := range r.b.Instances {
				if in.Params[0] == 0 && in.Transform1[3] == 3 {
					t.Fatalf("%s: a fill without clip coverage along the general path", c.Name)
				}
			}
			if d := differing(fast, general); d != 0 {
				t.Errorf("%s wide=%v: %d bytes differ between the fast and the general path", c.Name, wide, d)
			}
		}
	}
	r.b.GeneralFill, r.b.NoClipSkip = false, false
}
